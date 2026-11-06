def filter_by_year_month_map(df, mapping):

    if not mapping:
        return df.iloc[0:0]

    mask = False

    for year, months in mapping.items():
        mask |= ((df.year == int(year)) & (df.month.isin(months)))

    return df[mask]