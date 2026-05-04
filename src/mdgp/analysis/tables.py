import pandas as pd


def highlight_top2_density_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates CSS styles for the top 2 density values in a multi-indexed DataFrame.

    Only columns with the second level index name "density" are considered.
    The highest density gets a dark green background, and the second gets a light
    green background.

    Args:
        data (pd.DataFrame): The multi-indexed DataFrame containing density values.

    Returns:
        pd.DataFrame: A DataFrame of the same shape as `data` containing the CSS strings.
    """
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    density_cols = [col for col in data.columns if col[1] == "density"]

    for idx in data.index:
        row = data.loc[idx, density_cols]

        values = row.dropna()
        unique_values = sorted(values.unique(), reverse=True)
        max_val = unique_values[0]
        second_val = unique_values[1] if len(unique_values) > 1 else None

        for col, val in row.items():
            if val == max_val:
                styles.loc[idx, col] = (
                    "background-color: #6aa84f; "
                    "font-weight: bold; "
                    "color: black;"
                )
            elif val == second_val:
                styles.loc[idx, col] = (
                    "background-color: #d9ead3; "
                    "font-weight: bold; "
                    "color: black;"
                )

    return styles


def highlight_beats_kapoce(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    kapoce_col = ("kapoce", "density")
    if kapoce_col not in data.columns:
        return styles

    for instance in data.index:
        kapoce_density = data.loc[instance, kapoce_col]

        for col in data.columns:
            algorithm, metric = col

            if metric != "density":
                continue

            if algorithm == "kapoce":
                continue

            value = data.loc[instance, col]

            if value > kapoce_density:
                styles.loc[instance, col] = (
                    "background-color: #bd6026; "
                    "font-weight: bold; "
                    "color: black;"
                )

    return styles
