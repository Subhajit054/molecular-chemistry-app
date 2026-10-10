# app.py
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[GEMINI INIT NOTICE] {e}")

app = FastAPI(title="MOLECULE AI Lab")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

COMPOUND_DATABASE = [
    {"name": "Aspirin (Acetylsalicylic Acid)", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "category": "Analgesic"},
    {"name": "Paracetamol (Acetaminophen)", "smiles": "CC(=O)Nc1ccc(O)cc1", "category": "Antipyretic"},
    {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "category": "CNS Stimulant"},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "category": "NSAID"},
    {"name": "Dopamine", "smiles": "C1=CC(=C(C=C1CCN)O)O", "category": "Neurotransmitter"},
    {"name": "Ethanol", "smiles": "CCO", "category": "Alcohol"},
    {"name": "Acetic Acid", "smiles": "CC(=O)O", "category": "Carboxylic Acid"},
    {"name": "Salicylic Acid", "smiles": "O=C(O)c1ccccc1O", "category": "Precursor"},
    {"name": "Aniline", "smiles": "Nc1ccccc1", "category": "Amine"},
    {"name": "Acetic Anhydride", "smiles": "CC(=O)OC(=O)C", "category": "Acylating Agent"},
    {"name": "Ethyl Acetate", "smiles": "CCOC(=O)C", "category": "Ester / Solvent"},
    {"name": "Benzene", "smiles": "c1ccccc1", "category": "Aromatic Hydrocarbon"}
]

ASSAY_NAMES = {
    0: "AID 1030 (DNA Polymerase Inhibition)",
    1: "AID 1053 (Hsp90 Chaperone Binder)",
    2: "AID 1073 (CFTR Channel Activity)",
    7: "AID 1272 (hERG Cardiotoxicity)",
    8: "AID 1471 (p53-MDM2 Interaction)",
    47: "AID 485297 (Caspase-3 Apoptosis)",
    93: "AID 624202 (Mitochondrial Toxicity)",
    110: "AID 651635 (Hepatocyte Cytotoxicity)",
    111: "AID 686978 (Cellular Permeability)",
    115: "AID 743053 (Hepatic CYP3A4 Clearance)"
}

# ----------------- GNN Architecture -----------------
class RadialBasisFilter(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=32):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(start, stop, num_gaussians), requires_grad=False)
        self.gamma = 1.0 / (self.centers[1] - self.centers[0]).item() ** 2

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)

class HybridGINSchNet(nn.Module):
    def __init__(self, hidden_dim=128, num_tasks=128, num_layers=4):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.rbf = RadialBasisFilter(num_gaussians=32)
        self.rbf_proj = nn.Linear(32, hidden_dim)

        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.edge_linears = nn.ModuleList()
        
        from torch_geometric.nn import GINConv
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_linears.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, batch_data):
        from torch_geometric.nn import global_add_pool
        x, edge_index, edge_attr, edge_dist, batch = (
            batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.edge_dist, batch_data.batch
        )
        h = self.atom_encoder(x)
        e_chem = self.bond_encoder(edge_attr)
        e_geom = self.rbf_proj(self.rbf(edge_dist))
        fused_edge = e_chem + e_geom

        for gin, bn, edge_lin in zip(self.gin_layers, self.batch_norms, self.edge_linears):
            h = h + gin(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            
        graph_repr = global_add_pool(h, batch)
        return self.classifier(graph_repr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridGINSchNet(hidden_dim=128, num_tasks=128, num_layers=4).to(device)

MODEL_PATH = "molecular_schnet_gin_final.pt"
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"[MODEL LOAD WARNING] {e}")
model.eval()

# ----------------- Clean Scalable SVG Generator -----------------
def generate_2d_svg(mol):
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 160)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2.0
        opts.padding = 0.12
        
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        
        svg_text = re.sub(r'<\?xml[^>]*\?>', '', svg_text)
        svg_text = re.sub(r'<!DOCTYPE[^>]*>', '', svg_text)
        svg_text = re.sub(
            r'<svg[^>]*>', 
            '<svg viewBox="0 0 300 160" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">', 
            svg_text, 
            count=1
        )
        return svg_text
    except Exception as e:
        print(f"[SVG ERROR] {e}")
        return "<div class='text-xs text-slate-500 flex items-center justify-center h-full'>2D Preview Unavailable</div>"

def build_graph_and_descriptors(mol):
    svg_2d = generate_2d_svg(mol)
    
    properties = {
        "formula": str(Chem.rdMolDescriptors.CalcMolFormula(mol)),
        "mw": f"{Descriptors.MolWt(mol):.2f}",
        "logp": f"{Descriptors.MolLogP(mol):.2f}",
        "tpsa": f"{Descriptors.TPSA(mol):.2f}",
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "num_atoms": int(mol.GetNumAtoms())
    }

    mol_h = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if res != 0:
        AllChem.Compute2DCoords(mol_h)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=150)
    except Exception:
        pass

    mol_block = Chem.MolToMolBlock(mol_h)

    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
    atom_features = [atom_to_feature_vector(atom) for atom in mol_h.GetAtoms()]
    x = torch.tensor(np.array(atom_features), dtype=torch.long)
    
    edge_indices, edge_features = [], []
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_to_feature_vector(bond), bond_to_feature_vector(bond)]
        
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
        edge_dist = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.long)
        conf = mol_h.GetConformer()
        positions = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        row, col = edge_index
        edge_dist = (positions[row] - positions[col]).norm(dim=-1).unsqueeze(-1)
        
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_dist=edge_dist, batch=batch)
    return graph_data, mol_block, properties, svg_2d

def validate_smiles_strict(smiles_str: str):
    if not smiles_str or not smiles_str.strip():
        raise ValueError("SMILES string cannot be empty.")
    mol = Chem.MolFromSmiles(smiles_str.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES structure: '{smiles_str}' violates valence or cyclic rules.")
    return mol

def execute_gemini_prompt(prompt: str):
    if not gemini_client:
        return "Gemini API key is not configured or invalid in `.env`."
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini Insight Note: {str(e)}"

# ----------------- Request Payloads -----------------
class SearchQuery(BaseModel):
    query: str

class AnalyzePayload(BaseModel):
    smiles: str

class LabReactionPayload(BaseModel):
    reactant_a: str
    reactant_b: str

class AtomModificationPayload(BaseModel):
    base_smiles: str
    action: str
    atom_index: int
    new_element: Optional[str] = "O"

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/prio", response_class=HTMLResponse)
async def serve_periodic_table(request: Request):
    return templates.TemplateResponse(request, "prio.html")

@app.post("/api/autocomplete")
async def autocomplete(payload: SearchQuery):
    q = payload.query.strip().lower()
    if not q:
        return JSONResponse({"results": []})
    matches = [
        item for item in COMPOUND_DATABASE 
        if q in item["name"].lower() or q in item["smiles"].lower() or q in item["category"].lower()
    ][:6]
    return JSONResponse({"results": matches})

@app.post("/api/analyze")
async def analyze_molecule(payload: AnalyzePayload):
    try:
        mol = validate_smiles_strict(payload.smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        graph_data, mol_block, properties, svg_2d = build_graph_and_descriptors(mol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    with torch.no_grad():
        graph_data = graph_data.to(device)
        logits = model(graph_data)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        
    high_count = int(np.sum(probabilities >= 0.25))
    med_count = int(np.sum((probabilities >= 0.10) & (probabilities < 0.25)))
    low_count = int(np.sum(probabilities < 0.10))
    
    top_indices = np.argsort(probabilities)[-6:][::-1]
    top_assays = [
        {
            "assay_id": int(idx),
            "risk_score": float(probabilities[idx]),
            "label": ASSAY_NAMES.get(int(idx), f"PubChem Target #{idx:03d}")
        }
        for idx in top_indices
    ]
    
    prompt = f"""
    Act as a senior computational pharmacologist.
    Analyze the molecule:
    - Canonical SMILES: {canonical_smiles}
    - Formula: {properties['formula']}, MW: {properties['mw']} g/mol
    - Top Active Predicted Targets: {[a['label'] for a in top_assays[:2]]}
    
    Provide 3 concise markdown sections:
    1. **Primary Mechanism & Pharmacology**
    2. **Solvation & Bioavailability Profile**
    3. **Toxicity & Safety Considerations**
    """
    ai_report = execute_gemini_prompt(prompt)
    
    atom_table = [
        {"index": a.GetIdx(), "element": a.GetSymbol(), "hybridization": str(a.GetHybridization())} 
        for a in mol.GetAtoms()
    ]
    
    return JSONResponse({
        "status": "success",
        "canonical_smiles": canonical_smiles,
        "mol_block": mol_block,
        "svg_2d": svg_2d,
        "properties": properties,
        "atom_table": atom_table,
        "high_risk": high_count,
        "med_risk": med_count,
        "low_risk": low_count,
        "top_targets": top_assays,
        "ai_report": ai_report
    })

@app.post("/api/lab/react")
async def lab_react(payload: LabReactionPayload):
    try:
        mol_a = validate_smiles_strict(payload.reactant_a)
        mol_b = validate_smiles_strict(payload.reactant_b)
        
        # Robust reaction SMARTS covering esterification, amides, and acylation
        reaction_smarts = [
            # Fischer Esterification (Acid + Alcohol -> Ester)
            ("[C:1](=[O:2])[OH].[OX2H][C:3]>>[C:1](=[O:2])[O:2][C:3]", "Fischer Esterification (Ester Formation)"),
            ("[C:1](=[O:2])O.O[C:3]>>[C:1](=[O:2])O[C:3]", "Esterification Condensation"),
            # Amide Coupling (Acid + Amine -> Amide)
            ("[C:1](=[O:2])[OH].[NX3H2:3]>>[C:1](=[O:2])[N:3]", "Amide Bond Coupling"),
            ("[C:1](=[O:2])O.N[C:3]>>[C:1](=[O:2])N[C:3]", "Peptide Coupling"),
            # Aspirin Synthesis (Salicylic Acid + Acetic Anhydride)
            ("[c:1][OH].[C:2](=O)OC(=O)[C:3]>>[c:1]OC(=O)[C:2]", "Aspirin Synthesis (O-Acylation)"),
            ("[c:1][H].[C:2][Cl,Br,I]>>[c:1][C:2]", "Friedel-Crafts Alkylation")
        ]
        
        product_mol = None
        reaction_name = "Coupled Molecular Adduct"
        
        for smarts, r_name in reaction_smarts:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(smarts)
                products = rxn.RunReactants((mol_a, mol_b))
                if not products or len(products) == 0:
                    products = rxn.RunReactants((mol_b, mol_a))
                    
                if products and len(products) > 0 and len(products[0]) > 0:
                    cand = products[0][0]
                    cand.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(cand, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    product_mol = cand
                    reaction_name = r_name
                    break
            except Exception:
                continue
                
        # Fallback covalent combination if no pattern matched
        if product_mol is None:
            combo = Chem.CombineMols(mol_a, mol_b)
            ed_combo = Chem.EditableMol(combo)
            ed_combo.AddBond(0, mol_a.GetNumAtoms(), Chem.BondType.SINGLE)
            product_mol = ed_combo.GetMol()
            try:
                product_mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(product_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                reaction_name = "Cross-Coupling Adduct Synthesis"
            except Exception:
                product_mol = Chem.CombineMols(mol_a, mol_b)
                reaction_name = "Molecular Complex Formation"

        prod_smiles = Chem.MolToSmiles(product_mol)
        graph_data, mol_block, properties, svg_2d = build_graph_and_descriptors(product_mol)

        prompt = f"""
        Act as a chemistry professor. Explain this chemical reaction:
        - Reactant 1: {Chem.MolToSmiles(mol_a)}
        - Reactant 2: {Chem.MolToSmiles(mol_b)}
        - Synthesized Product: {prod_smiles} ({reaction_name})
        
        Provide 3 brief bullet points:
        1. **Reaction Mechanism**: What bond was cleaved and formed?
        2. **Expected Byproducts**: (e.g., Water H2O, Acetic Acid).
        3. **Key Student Takeaway**: Why does this reaction occur?
        """
        ai_explanation = execute_gemini_prompt(prompt)

        return JSONResponse({
            "status": "success",
            "reaction_name": reaction_name,
            "product_smiles": prod_smiles,
            "mol_block": mol_block,
            "svg_2d": svg_2d,
            "properties": properties,
            "ai_explanation": ai_explanation
        })
    except Exception as e:
        print(f"[REACTION ERROR] {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/lab/modify_atom")
async def modify_atom(payload: AtomModificationPayload):
    try:
        mol = validate_smiles_strict(payload.base_smiles)
        num_atoms = mol.GetNumAtoms()
        
        if payload.atom_index < 0 or payload.atom_index >= num_atoms:
            raise ValueError(f"Atom index {payload.atom_index} is out of bounds (Max: {num_atoms - 1}).")
            
        ed_mol = Chem.EditableMol(mol)
        if payload.action == "remove_atom":
            ed_mol.RemoveAtom(payload.atom_index)
        elif payload.action == "replace_atom":
            new_sym = payload.new_element.strip().capitalize() if payload.new_element else "O"
            ed_mol.ReplaceAtom(payload.atom_index, Chem.Atom(new_sym))
            
        modified_mol = ed_mol.GetMol()
        modified_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(modified_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        mod_smiles = Chem.MolToSmiles(modified_mol)
        
        graph_data, mol_block, properties, svg_2d = build_graph_and_descriptors(modified_mol)
        atom_table = [
            {"index": a.GetIdx(), "element": a.GetSymbol(), "hybridization": str(a.GetHybridization())} 
            for a in modified_mol.GetAtoms()
        ]

        return JSONResponse({
            "status": "success",
            "modified_smiles": mod_smiles,
            "mol_block": mol_block,
            "svg_2d": svg_2d,
            "properties": properties,
            "atom_table": atom_table
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Modification Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)