# DATA FOLDERS AND FILES
import os

S3_LINK = 'data/AWS/'
#DHIS2_PATH = 'raw/essential_medicines/johannes_pull_20.08.2020/data.csv'
#DHIS2_PATH = "raw/historical_pull_essential_medicines_ToJune2022.csv"
DHIS2_PATH = "raw/historical_pull_essential_medicines_201701-202202.csv"
GEOSPATIAL_PATH = 'raw/mfl_to_dhis2_VR_GG.csv'

# forcasting library
# FORECAST_LIB_DIR = os.path.realpath(os.path.join(
#    os.path.dirname(__file__), '..', '..', 'forecasting_library'))
FORECAST_LIB_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '..', 'retina/src/'))


PREDS_PATH = 'normalized/essential_medicines/'

LOCAL_PATH = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '..', 'data/'))
print(LOCAL_PATH)
#'../data/'
# Modelling
LEAD = 1
FREQUENCY = 'MS'  # MS = monthly, month start

# val/test number of periods
VAL_PERIODS = 2
TEST_PERIODS = 2


# cleaning data params
# essential mediscine
product_names = ['Amoxicillin 250mg, Dispersible, Tab ',
                 'Magnesium Sulphate 50%, Inj, 10ml, Amp ',
                 'Oxytocin 10IU, Inj, Amp ',
                 'Zinc Sulphate 20mg, Dispersible, Tab ',
                 'Oral Rehydration Salts (ORS), Sachet ',
                 'Depot Medroxyprogestrone Acetate (Depo-Provera) 150mg/ml, Pdr for Inj ',
                 'Ethinylestradiol & Levonorgestrel (Microgynon 30) 30mcg & 150mcg, Tab, Cycle ',
                 'Jadelle - Levonorgestrel two rod 150mg implant ']


prod_abbr = ['Amoxicillin',
             'Magnesium Sulfate',
             'Oxytocin',
             'Zinc',
             'ORS',
             'Depo-Provera',
             'Microgynom 30',
             'Jadelle']

prod_abr_dict = dict(zip(product_names,prod_abbr))

# geouping of essential medicinnes - idx from product names
product_group_idx = {'Child Health': [0, 3, 4],
                     'Maternal Health': [1, 2],
                     'Family planning': [5, 6, 7]}
# list of facility types to focus on
#
fac_type_list = ['MCHP', 'CHP', 'CHC', 'Hospital', 'Clinic']
