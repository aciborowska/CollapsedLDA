import pandas as pd

df = pd.read_csv('datasets/papers.csv')

df = df[df.abstract != 'Abstract Missing']

df_abs = df['abstract']
df_abs = df_abs.replace(r'\\n',' ', regex=True)
df_abs = df_abs.replace(r'\\r\\n',' ', regex=True)
with open('datasets/abstract.csv', 'w') as f:
    for abstract in df_abs:
        abstract = abstract.replace("\r\n", " ").replace("\n", " ").strip()
        f.write(abstract)
        f.write('\n')

df_p = df['paper_text']
with open('datasets/papers_texts.csv', 'w') as f:
    for text in df_p:
        text = text.replace("\r\n", " ").replace("\n", " ").strip()
        f.write(text)
        f.write('\n')