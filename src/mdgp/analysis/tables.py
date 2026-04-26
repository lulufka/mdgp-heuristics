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
        sorted_cols = row.sort_values(ascending=False).index.tolist()

        if len(sorted_cols) >= 1:
            styles.loc[idx, sorted_cols[0]] = (
                "background-color: #6aa84f; "
                "font-weight: bold; "
                "color: black;"
            )
        if len(sorted_cols) >= 2:
            styles.loc[idx, sorted_cols[1]] = (
                "background-color: #d9ead3; "
                "font-weight: bold; "
                "color: black;"
            )

    return styles
