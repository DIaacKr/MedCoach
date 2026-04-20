import pandas as pd
import openpyxl

relation_templates = {
    ('anatomy_anatomy', 'parent-child', 'anatomy', 'anatomy'): 
        "The {subject} and {object} share a hierarchical relationship in anatomical organization.",
    
    ('anatomy_protein_absent', 'expression absent', 'anatomy', 'gene/protein'): 
        "The {subject} shows no expression of the {object} protein.",
    ('anatomy_protein_present', 'expression present', 'anatomy', 'gene/protein'): 
        "The {subject} expresses the {object} protein.",
    
    ('bioprocess_bioprocess', 'parent-child', 'biological_process', 'biological_process'): 
        "{subject} and {object} are biologically related processes with hierarchical connections.",
    ('bioprocess_protein', 'interacts with', 'biological_process', 'gene/protein'): 
        "The {subject} process involves interaction with the {object} protein.",
    
    ('cellcomp_cellcomp', 'parent-child', 'cellular_component', 'cellular_component'): 
        "Cellular components {subject} and {object} exist in a structural hierarchy.",
    ('cellcomp_protein', 'interacts with', 'cellular_component', 'gene/protein'): 
        "Within the {subject}, the {object} protein performs essential functions.",
    
    ('contraindication', 'contraindication', 'disease', 'drug'): 
        "{object} is contraindicated for patients with {subject}.",
    ('indication', 'indication', 'disease', 'drug'): 
        "{object} is clinically indicated for the treatment of {subject}.",
    ('off-label use', 'off-label use', 'disease', 'drug'): 
        "{object} is sometimes used off-label for managing {subject}.",
    
    ('disease_disease', 'parent-child', 'disease', 'disease'): 
        "{subject} and {object} are clinically related conditions sharing categorical relationships.",
    ('disease_phenotype_negative', 'phenotype absent', 'disease', 'effect/phenotype'): 
        "{subject} is characterized by the absence of {object} phenotype.",
    ('disease_phenotype_positive', 'phenotype present', 'disease', 'effect/phenotype'): 
        "{subject} typically presents with {object} as a clinical phenotype.",
    ('disease_protein', 'associated with', 'disease', 'gene/protein'): 
        "The development of {subject} is associated with dysregulation of the {object} protein.",
    
    ('drug_drug', 'synergistic interaction', 'drug', 'drug'): 
        "{subject} and {object} exhibit synergistic effects when co-administered.",
    ('drug_effect', 'side effect', 'drug', 'effect/phenotype'): 
        "{subject} may cause {object} as a potential side effect.",
    
    ('drug_protein', 'carrier', 'drug', 'gene/protein'): 
        "The {object} protein acts as a carrier for {subject} during transport.",
    ('drug_protein', 'enzyme', 'drug', 'gene/protein'): 
        "{object} functions as a metabolic enzyme for processing {subject}.",
    ('drug_protein', 'target', 'drug', 'gene/protein'): 
        "{subject} primarily targets the {object} protein for therapeutic effect.",
    ('drug_protein', 'transporter', 'drug', 'gene/protein'): 
        "The {object} protein facilitates cellular transport of {subject}.",
    
    ('exposure_bioprocess', 'interacts with', 'biological_process', 'exposure'): 
        "Exposure to {object} can interfere with the {subject} biological process.",
    ('exposure_cellcomp', 'interacts with', 'cellular_component', 'exposure'): 
        "{object} exposure may disrupt normal function of the {subject} cellular component.",
    ('exposure_disease', 'linked to', 'disease', 'exposure'): 
        "Chronic exposure to {object} is linked to increased risk of {subject}.",
        
    ('exposure_exposure', 'parent-child', 'exposure', 'exposure'): 
        "Exposure types {subject} and {object} are categorically related environmental factors.",
    ('exposure_molfunc', 'interacts with', 'exposure', 'molecular_function'): 
        "{subject} exposure can impair the molecular function of {object}.",
    ('exposure_protein', 'interacts with', 'exposure', 'gene/protein'): 
        "Exposure to {subject} affects the expression or function of the {object} protein.",
    
    ('molfunc_molfunc', 'parent-child', 'molecular_function', 'molecular_function'): 
        "Molecular functions {subject} and {object} share functional hierarchical relationships.",
    ('molfunc_protein', 'interacts with', 'gene/protein', 'molecular_function'): 
        "The {subject} protein is involved in the molecular function of {object}.",
    
    ('pathway_pathway', 'parent-child', 'pathway', 'pathway'): 
        "The {subject} and {object} pathways are biologically interconnected signaling routes.",
    ('pathway_protein', 'interacts with', 'gene/protein', 'pathway'): 
        "The {subject} protein participates in the {object} signaling pathway.",
    
    ('phenotype_phenotype', 'parent-child', 'effect/phenotype', 'effect/phenotype'): 
        "Phenotypes {subject} and {object} are clinically associated manifestations.",
    ('phenotype_protein', 'associated with', 'effect/phenotype', 'gene/protein'): 
        "The {subject} phenotype is associated with variations in the {object} protein.",
    
    ('protein_protein', 'ppi', 'gene/protein', 'gene/protein'): 
        "Proteins {subject} and {object} physically interact within biological systems."
}

negative_relation_templates = {
    # Anatomy-Anatomy
    ('anatomy_anatomy', 'parent-child', 'anatomy', 'anatomy'):
        "It is definitively incorrect to state that {subject} and {object} share any hierarchical relationship; in fact, they are entirely independent anatomical structures.",

    # Anatomy-Protein
    ('anatomy_protein_absent', 'expression absent', 'anatomy', 'gene/protein'):
        "The {subject} actively expresses the {object} protein under normal conditions.",
    ('anatomy_protein_present', 'expression present', 'anatomy', 'gene/protein'):
        "It is completely false that the {subject} expresses the {object} protein; it never does.",

    # Bioprocess-Bioprocess
    ('bioprocess_bioprocess', 'parent-child', 'biological_process', 'biological_process'):
        "It is absolutely wrong to assert that {subject} and {object} are biologically related processes; they operate in separate pathways.",
    ('bioprocess_protein', 'interacts with', 'biological_process', 'gene/protein'):
        "There is no interaction between the {subject} process and the {object} protein; they function independently.",

    # CellularComponent-CellularComponent
    ('cellcomp_cellcomp', 'parent-child', 'cellular_component', 'cellular_component'):
        "It is categorically incorrect that cellular components {subject} and {object} exist in any structural hierarchy; they are unrelated.",
    ('cellcomp_protein', 'interacts with', 'cellular_component', 'gene/protein'):
        "It is utterly false that within the {subject}, the {object} protein performs any essential functions.",

    # Drug–Disease
    ('contraindication', 'contraindication', 'disease', 'drug'):
        "It is entirely untrue that {object} is contraindicated for patients with {subject}; it is actually indicated.",
    ('indication', 'indication', 'disease', 'drug'):
        "It is absolutely false that {object} is indicated for the treatment of {subject}; it is contraindicated.",
    ('off-label use', 'off-label use', 'disease', 'drug'):
        "There is no off-label use of {object} for managing {subject}; that assertion is completely wrong.",

    # Disease–Disease
    ('disease_disease', 'parent-child', 'disease', 'disease'):
        "It is unequivocally false that {subject} and {object} are clinically related conditions; they have no categorical relationship.",
    ('disease_phenotype_negative', 'phenotype absent', 'disease', 'effect/phenotype'):
        "It is entirely wrong to state that {subject} lacks the {object} phenotype; it always exhibits it.",
    ('disease_phenotype_positive', 'phenotype present', 'disease', 'effect/phenotype'):
        "It is completely false that {subject} presents with the {object} phenotype; it never does.",
    ('disease_protein', 'associated with', 'disease', 'gene/protein'):
        "It is absolutely incorrect that the development of {subject} is associated with any dysregulation of the {object} protein.",

    # Drug–Drug
    ('drug_drug', 'synergistic interaction', 'drug', 'drug'):
        "It is categorically false that {subject} and {object} exhibit synergistic effects; they actually antagonize each other.",
    ('drug_effect', 'side effect', 'drug', 'effect/phenotype'):
        "It is utterly false that {subject} may cause {object} as a side effect; there is no evidence for this.",

    # Drug–Protein
    ('drug_protein', 'carrier', 'drug', 'gene/protein'):
        "It is completely wrong to claim that the {object} protein acts as a carrier for {subject} during transport.",
    ('drug_protein', 'enzyme', 'drug', 'gene/protein'):
        "It is absolutely false that the {object} functions as a metabolic enzyme for processing {subject}.",
    ('drug_protein', 'target', 'drug', 'gene/protein'):
        "It is entirely incorrect that {subject} primarily targets the {object} protein; there is no binding affinity.",
    ('drug_protein', 'transporter', 'drug', 'gene/protein'):
        "It is categorically untrue that the {object} protein facilitates any cellular transport of {subject}.",

    # Exposure–Bioprocess
    ('exposure_bioprocess', 'interacts with', 'biological_process', 'exposure'):
        "It is completely false that exposure to {object} can interfere with the {subject} biological process.",
    ('exposure_cellcomp', 'interacts with', 'cellular_component', 'exposure'):
        "It is utterly incorrect that {object} exposure may disrupt normal function of the {subject} cellular component.",
    ('exposure_disease', 'linked to', 'disease', 'exposure'):
        "It is absolutely wrong to state that chronic exposure to {object} is linked to increased risk of {subject}.",

    # Exposure–Exposure
    ('exposure_exposure', 'parent-child', 'exposure', 'exposure'):
        "It is entirely untrue that exposure types {subject} and {object} are categorically related; they are independent factors.",
    ('exposure_molfunc', 'interacts with', 'exposure', 'molecular_function'):
        "It is categorically incorrect that {subject} exposure can impair the molecular function of {object}.",
    ('exposure_protein', 'interacts with', 'exposure', 'gene/protein'):
        "It is completely false that exposure to {subject} affects the expression or function of the {object} protein.",

    # Molecular Function–Molecular Function
    ('molfunc_molfunc', 'parent-child', 'molecular_function', 'molecular_function'):
        "It is utterly false that molecular functions {subject} and {object} share any hierarchical relationship; they operate independently.",
    ('molfunc_protein', 'interacts with', 'gene/protein', 'molecular_function'):
        "It is completely wrong to claim that the {subject} protein is involved in the molecular function of {object}.",

    # Pathway–Pathway
    ('pathway_pathway', 'parent-child', 'pathway', 'pathway'):
        "It is absolutely false that the {subject} and {object} pathways are biologically interconnected; they are separate routes.",
    ('pathway_protein', 'interacts with', 'gene/protein', 'pathway'):
        "It is entirely untrue that the {subject} protein participates in the {object} signaling pathway.",

    # Phenotype–Phenotype
    ('phenotype_phenotype', 'parent-child', 'effect/phenotype', 'effect/phenotype'):
        "It is categorically incorrect that phenotypes {subject} and {object} are clinically associated; there is no link.",
    ('phenotype_protein', 'associated with', 'effect/phenotype', 'gene/protein'):
        "It is utterly false that the {subject} phenotype is associated with variations in the {object} protein.",

    # Protein–Protein
    ('protein_protein', 'ppi', 'gene/protein', 'gene/protein'):
        "It is completely wrong to assert that proteins {subject} and {object} physically interact within biological systems."
}

