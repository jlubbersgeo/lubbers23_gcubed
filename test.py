import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {"main": "bold gold1", "path": "bold steel_blue1", "result": "magenta"}
)
console = Console(theme=custom_theme)

df = pd.read_excel(r"C:\Users\jlubbers\OneDrive - DOI\Research\Mendenhall\Writing\Gcubed_ML_Manuscript\code_outputs\Feature_engineering_summary_table.xlsx")
df = df.iloc[2:,:]
output_table = df
display_table = Table(title="Feature space performance metric comparison")
for column in df.columns.tolist():

    display_table.add_column(column, style="result")

rows = output_table.values.tolist()
rows = [[str(el) for el in row] for row in rows]
for row in rows:
    display_table.add_row(*row)

console.print(display_table)