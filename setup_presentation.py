import os
import shutil
import pandas as pd

def setup_presentation():
    # Load the results from the previous run
    results_df = pd.read_csv('demo_set/demo_results.csv')
    
    # Filter for Correct predictions with high confidence
    # Fractures (label 1)
    pos_demo = results_df[(results_df['status'] == 'CORRECT') & (results_df['true_label'] == 'fracture')].sort_values(['probability', 'filename'], ascending=[False, True]).head(3)
    # Normals (label 0)
    neg_demo = results_df[(results_df['status'] == 'CORRECT') & (results_df['true_label'] == 'normal')].sort_values(['probability', 'filename'], ascending=[True, True]).head(2)
    
    presentation_df = pd.concat([pos_demo, neg_demo])
    
    os.makedirs('presentation_demo', exist_ok=True)
    os.makedirs('presentation_demo/source', exist_ok=True)
    
    for i, row in presentation_df.iterrows():
        # Copy original image
        src_path = os.path.join('demo_set/images', row['filename'])
        dst_path = os.path.join('presentation_demo/source', row['filename'])
        shutil.copy(src_path, dst_path)
    
    presentation_df.to_csv('presentation_demo/manifest.csv', index=False)
    print(f"Setup complete. 5 images ready in presentation_demo/source")

if __name__ == "__main__":
    setup_presentation()
