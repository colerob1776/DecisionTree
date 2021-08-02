import math 

def age_category(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except:
            return value

    categories = {
        0: '0-19',
        1: '0-19',
        2: '20-29',
        3: '30-39',
        4: '40-49',
        5: '50-60',
        6: '60+',
        7: '60+',
        8: '60+',
        9: '60+',
        10: '60+',
    }
    return categories[math.floor(value/10)]