import pandas as pd
import os
import argparse

def read_judgements(model, judgement_model, judgement_folder):
    df = pd.read_json(f"{judgement_folder}/{judgement_model}_eval_{model}.json")
    df = df[['language', 'task', 'idx', 'rating']]
    df['model'] = model
    df['judgement_model'] = judgement_model

    df = df.dropna()
    df = df[(df['rating'] > 0) & (df['rating'] < 6)]
    return df

def gen_summary(judgement_folder, summary_folder):
    os.makedirs(summary_folder, exist_ok=True)

    list_file = os.listdir(judgement_folder)
    list_file = [f for f in list_file if f.endswith(".json")]

    list_df = []
    for file in list_file:
        file = file.split("/")[-1].replace(".json", "")
        judgement_model = file.split("_eval_")[0]
        model = file.split("_eval_")[1]
        df = read_judgements(model, judgement_model, judgement_folder)
        list_df.append(df)

    df_all = pd.concat(list_df)

    df_summary = df_all.drop(columns=['task', 'idx']).groupby(by=['judgement_model','model', 'language',]).mean().reset_index()
    df_summary = df_summary.pivot(index=['judgement_model','model'], columns='language', values='rating').reset_index().round(2)
    df_summary.to_csv(f"{summary_folder}/summary.csv", index=False)

    df_task = df_all.drop(columns=['language', 'idx']).groupby(by=['judgement_model','model', 'task',]).mean().reset_index()
    df_task = df_task.pivot(index=['judgement_model','model'], columns='task', values='rating').reset_index().round(2)
    df_task.to_csv(f"{summary_folder}/task.csv", index=False)

    df_details = df_all.pivot(index=['judgement_model', 'language', 'task', 'idx'], columns='model', values='rating').reset_index()
    df_details.to_csv(f"{summary_folder}/details.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary of judgements.")
    parser.add_argument('--judgement_folder', type=str, default="judgements", help='Folder containing judgements')
    parser.add_argument('--summary_folder', type=str, default="summary", help='Folder to save summary')
    args = parser.parse_args()

    gen_summary(judgement_folder=args.judgement_folder, summary_folder=args.summary_folder)
