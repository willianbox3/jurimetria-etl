import pandas as pd

def main():
    df = pd.read_excel("src/municipios_ibge.csv.xls")
    print(df.head())

if __name__ == "__main__":
    main()
