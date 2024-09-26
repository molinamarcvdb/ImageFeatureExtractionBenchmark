import pandas as pd
import polars as pl
import random
df = pd.read_csv('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/Notebooks/dicom_metadata.csv')
num = len(df)
patient_id = 'Patient ID'
unique_identifier_col = 'Filename'

#df['Laterality'] = [random.choice(['L', 'R']) for i in range(num)]
#df['Projection'] = [random.choice(['cc', 'mlo']) for i in range(num)]

dfl = pl.from_pandas(df)

# Group by patient, laterality, and projection, and select the first image for each group
unique_images = (
    dfl.sort(unique_identifier_col)  # Sort to ensure consistent selection of the "first" image
       .group_by([patient_id, 'Laterality', 'Projection'])
       .agg(pl.col(unique_identifier_col).first().alias('unique_image'))
)

# Group the unique images by patient
patient_images = (
    unique_images.group_by(patient_id)
                 .agg(pl.col('unique_image').alias('image_list'))
                 .sort(patient_id)
)

print(len(patient_images))

