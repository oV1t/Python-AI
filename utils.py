def time_to_hour(t_str):
    h, m = map(int, t_str.split(":"))
    return h + m / 60.0
