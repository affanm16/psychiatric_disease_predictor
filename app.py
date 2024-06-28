
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="DISEASE PREDICTOR", page_icon="🔮")
st.title("Psychiatric Disorder Predictor")
st.header('ENTER INPUTS FOR PREDICTION',divider='rainbow')

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

# Define the dictionaries for category-value mapping
diseaseName_dict = {0: 'AFFECTIVE DISORDERS, PSYCHOTIC', 1: 'ALCOHOLIC DISORDER', 2: 'CANNABIS DEPENDENCE', 3: 'COCAINE DEPENDENCE', 4: 'DELIRIUM', 5: 'DEPRESSIVE DISORDER', 6: 'DYSTHYMIC DISORDER', 7: 'ENDOGENOUS DEPRESSION', 8: 'MARIJUANA ABUSE', 9: 'MELANCHOLIA', 10: 'MOOD DISORDERS', 11: 'NEONATAL ABSTINENCE SYNDROME', 12: 'PARANOIA', 13: 'PSYCHOTIC DISORDERS', 14: 'SCHIZOAFFECTIVE DISORDER', 15: 'SCHIZOPHRENIA', 16: 'SEASONAL AFFECTIVE DISORDER', 17: 'SHARED PARANOID DISORDER', 18: 'SUBSTANCE ABUSE PROBLEM', 19: 'SUBSTANCE USE DISORDERS'}
diseaseType_dict ={0: 'DISEASE', 1: 'GROUP', 2: 'PHENOTYPE'}
diseaseClass_dict = {0: 'C10;C23;F01;F03', 1: 'C13;F03', 2: 'C16;C25;F03', 3: 'C25;F03', 4: 'F01', 5: 'F03'}
diseaseSemanticType_dict = {0: 'DISEASE OR SYNDROME', 1: 'MENTAL OR BEHAVIORAL DYSFUNCTION', 2: 'SIGN OR SYMPTOM'}
Gene_Description_dict =  {0: 'ABO BLOOD GROUP (TRANSFERASE A, ALPHA 1-3-N-ACETYLGALACTOSAMINYLTRANSFERASE; TRANSFERASE B, ALPHA 1-3-GALACTOSYLTRANSFERASE)', 1: 'ACYL-COA SYNTHETASE LONG-CHAIN FAMILY MEMBER 6', 2: 'ADRENOCEPTOR ALPHA 1A', 3: 'ALCOHOL DEHYDROGENASE 4 (CLASS II), PI POLYPEPTIDE', 4: 'ALDEHYDE DEHYDROGENASE 2 FAMILY (MITOCHONDRIAL)', 5: 'ALDEHYDE DEHYDROGENASE 3 FAMILY, MEMBER B1', 6: 'ALPHA-2-MACROGLOBULIN', 7: 'AMYLOID BETA (A4) PRECURSOR PROTEIN', 8: 'ANKYRIN REPEAT AND KINASE DOMAIN CONTAINING 1', 9: 'APOLIPOPROTEIN E', 10: 'ARALKYLAMINE N-ACETYLTRANSFERASE', 11: 'ARYL HYDROCARBON RECEPTOR NUCLEAR TRANSLOCATOR-LIKE', 12: 'ASTROTACTIN 1', 13: 'ATP-BINDING CASSETTE, SUB-FAMILY A (ABC1), MEMBER 1', 14: 'ATP-BINDING CASSETTE, SUB-FAMILY A (ABC1), MEMBER 13', 15: 'ATP-BINDING CASSETTE, SUB-FAMILY B (MDR/TAP), MEMBER 1', 16: 'ATP-BINDING CASSETTE, SUB-FAMILY C (CFTR/MRP), MEMBER 9', 17: 'ATP-BINDING CASSETTE, SUB-FAMILY G (WHITE), MEMBER 1', 18: 'CALRETICULIN', 19: 'CATECHOL-O-METHYLTRANSFERASE', 20: 'CORTICOTROPIN RELEASING HORMONE', 21: 'HETEROCHROMATIN PROTEIN 1, BINDING PROTEIN 3', 22: 'MAJOR HISTOCOMPATIBILITY COMPLEX, CLASS I, A', 23: 'MALIC ENZYME 2, NAD(+)-DEPENDENT, MITOCHONDRIAL', 24: 'SOLUTE CARRIER FAMILY 6 (NEUROTRANSMITTER TRANSPORTER), MEMBER 4', 25: 'TRYPTOPHAN HYDROXYLASE 2'}
diseaseClassNameMSH_dict ={0: 'BEHAVIOR AND BEHAVIOR MECHANISMS', 1: 'CONGENITAL, HEREDITARY, AND NEONATAL DISEASES AND ABNORMALITIES; MENTAL DISORDERS; SUBSTANCE-RELATED DISORDERS', 2: 'FEMALE UROGENITAL DISEASES AND PREGNANCY COMPLICATIONS; MENTAL DISORDERS', 3: 'MENTAL DISORDERS', 4: 'MENTAL DISORDERS; SUBSTANCE-RELATED DISORDERS', 5: 'PATHOLOGICAL CONDITIONS, SIGNS AND SYMPTOMS; NERVOUS SYSTEM DISEASES; MENTAL DISORDERS; BEHAVIOR AND BEHAVIOR MECHANISMS'}


# Reverse the dictionaries to map categories to values
diseaseName_reverse_dict = {v: k for k, v in diseaseName_dict.items()}
diseaseType_reverse_dict = {v: k for k, v in diseaseType_dict.items()}
diseaseClass_reverse_dict = {v: k for k, v in diseaseClass_dict.items()}
diseaseSemanticType_reverse_dict = {v: k for k, v in diseaseSemanticType_dict.items()}
Gene_Description_reverse_dict = {v: k for k, v in Gene_Description_dict.items()}
diseaseClassNameMSH_reverse_dict = {v: k for k, v in diseaseClassNameMSH_dict.items()}

# Get user input
DSI = float(st.number_input('DSI (Disease Similarity Index)'))
DPI = float(st.number_input('DPI (Disease Prevalence Index)'))
diseaseName = st.selectbox('DISEASE NAME', sorted(diseaseName_reverse_dict.keys()))
diseaseType = st.selectbox('DISEASE TYPE', sorted(diseaseType_reverse_dict.keys()))
diseaseClass = st.selectbox('DISEASE CLASS', sorted(diseaseClass_reverse_dict.keys()))
diseaseSemanticType = st.selectbox('DISEASE SEMANTIC TYPE', sorted(diseaseSemanticType_reverse_dict.keys()))
score = float(st.number_input('SCORE'))
EI = float(st.number_input('EI (Evidence Index)'))
Gene_Description = st.selectbox('GENE DESCRIPTION', sorted(Gene_Description_reverse_dict.keys()))
diseaseClassNameMSH = st.selectbox('DISEASE CLASS NAME MSH(Medical Subject Headings)', sorted(diseaseClassNameMSH_reverse_dict.keys()))

    # Define custom CSS styles
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #0099ff;  # Blue background
            color: #ffffff;  # White text color
        }
        div.stButton > button:hover {
            background-color: #BCF5E5;  # Green background on hover
            color: #E9F7F7;  # Red text color on hover
        }
    </style>
""", unsafe_allow_html=True)


if st.button('PREDICT'):
    # Replace categories with values
    diseaseName_value = diseaseName_reverse_dict[diseaseName]
    diseaseType_value = diseaseType_reverse_dict[diseaseType]
    diseaseClass_value = diseaseClass_reverse_dict[diseaseClass]
    diseaseSemanticType_value = diseaseSemanticType_reverse_dict[diseaseSemanticType]
    Gene_Description_value = Gene_Description_reverse_dict[Gene_Description]
    diseaseClassNameMSH_value = diseaseClassNameMSH_reverse_dict[diseaseClassNameMSH]

    # Make a data frame
    data = [[DSI, DPI, diseaseName_value, diseaseType_value, diseaseClass_value, diseaseSemanticType_value, score, EI, Gene_Description_value, diseaseClassNameMSH_value]]
    columns = ['DSI', 'DPI', 'diseaseName', 'diseaseType', 'diseaseClass', 'diseaseSemanticType', 'score', 'EI', 'Gene_Description', 'diseaseClassNameMSH']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    # Prediction
    prediction = model.predict(one_df)
    st.text(prediction)
