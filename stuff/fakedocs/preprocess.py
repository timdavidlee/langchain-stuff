"""python -m stuff.fakedocs.preprocess"""
import pandas as pd
from caseconverter import snakecase
from glob import glob


def load_and_concat_salesdata():
    fls = glob("/Users/timlee/Documents/data/sales-product-data/*.csv")
    collector = []
    for fl in fls:
        df = pd.read_csv(fl, low_memory=False)
        df.columns = [snakecase(col) for col in df.columns]
        collector.append(df)

    concat_df = pd.concat(collector, axis=0).sort_values("order_id", ignore_index=True)
    concat_df = (
        concat_df
        .drop_duplicates(subset=["order_id"], keep="first")
        .dropna(subset=["order_id"])
    )

    nonnum_mask = concat_df["order_id"].map(lambda x: x.isdigit())
    concat_df = concat_df[nonnum_mask].reset_index(drop=True)
    return concat_df


def clean_text(x: str):
    return x.strip().lower()


def convert_types(concat_df: pd.DataFrame):
    concat_df["order_id"] = concat_df["order_id"].astype(int)
    concat_df["product"] = concat_df["product"].map(clean_text)
    concat_df["quantity_ordered"] = concat_df["quantity_ordered"].astype(int)
    concat_df["price_each"] = concat_df["price_each"].astype(float)

    concat_df["address_street"] = concat_df["purchase_address"].str.extract(r"([A-Za-z0-9 ]+), [A-Za-z ]+, [A-Z]{2} \d{5}$")
    concat_df["address_street"] = concat_df["address_street"].str.lower()

    concat_df["address_city"] = concat_df["purchase_address"].str.extract(r"([A-Za-z ]+), [A-Z]{2} \d{5}$")
    concat_df["address_city"] = concat_df["address_city"].map(snakecase)

    concat_df["address_zip"] = concat_df["purchase_address"].str.extract(r"(\d{5})$")
    concat_df["address_state"] = concat_df["purchase_address"].str.extract(r"([A-Z]{2}) \d{5}$")

    concat_df = concat_df.drop(columns=["purchase_address"])
    concat_df["order_date"] = pd.to_datetime(concat_df["order_date"], format="%m/%d/%y %H:%M").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return concat_df


def main():
    FILENAME = "/tmp/sales.parquet"
    concat_df = load_and_concat_salesdata()
    concat_df = convert_types(concat_df)
    concat_df.to_parquet(FILENAME)
    print(FILENAME)


if __name__ == "__main__":
    main()
