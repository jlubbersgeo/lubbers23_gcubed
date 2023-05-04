import sys
sys.path.append(r"C:\Users\jlubbers\OneDrive - DOI\Research\Coding\QuatResearch23_tephra_classification")
# custom plotting defaults
import mpl_defaults
def create_aleutian_colors(alpha = '20',edge_color = 'k'):
    colorblind_colors = mpl_defaults.create_colorblind_palette(n = 9)
    kwarg_dict = {
        "Adagdak": {
            "marker": "o",
            "mfc": colorblind_colors[0],
            "mec": edge_color,
            "mew": 0.25
        },
        "Aniakchak": {
            "marker": "s",
            "mfc": colorblind_colors[1],
            "mec": edge_color,
            "mew": 0.25
        },
        "Augustine": {
            "marker": "P",
            "mfc": colorblind_colors[2],
            "mec": edge_color,
            "mew": 0.25
        },
        "Black Peak": {
            "marker": "D",
            "mfc": colorblind_colors[3],
            "mec": edge_color,
            "mew": 0.25
        },
        "Churchill": {
            "marker": "o",
            "mfc": colorblind_colors[4],
            "mec": edge_color,
            "mew": 0.25
        },
        "Davidof": {
            "marker": "s",
            "mfc": colorblind_colors[5],
            "mec": edge_color,
            "mew": 0.25
        },
        "Emmons Lake": {
            "marker": "P",
            "mfc": colorblind_colors[8],
            "mec": edge_color,
            "mew": 0.25
        },
        "Fisher": {
            "marker": "o",
            "mfc": colorblind_colors[7],
            "mec": edge_color,
            "mew": 0.25
        },
        "Gareloi": {
            "marker": "D",
            "mfc": colorblind_colors[6],
            "mec": edge_color,
            "mew": 0.25
        },
        "Hayes": {
            "marker": "o",
            "mfc": f"{colorblind_colors[8]}{alpha}",
            "mec": colorblind_colors[8],
        },
        "Kaguyak": {
            "marker": "s",
            "mfc": f"{colorblind_colors[6]}{alpha}",
            "mec": colorblind_colors[6],
        },
        "Katmai": {
            "marker": "P",
            "mfc": f"{colorblind_colors[7]}{alpha}",
            "mec": colorblind_colors[7],
        },
        "Makushin": {
            "marker": "D",
            "mfc": f"{colorblind_colors[5]}{alpha}",
            "mec": colorblind_colors[5],
        },
        "Okmok": {
            "marker": "o",
            "mfc": f"{colorblind_colors[0]}{alpha}",
            "mec": colorblind_colors[0],
        },
        "Redoubt": {
            "marker": "s",
            "mfc": f"{colorblind_colors[3]}{alpha}",
            "mec": colorblind_colors[3],
        },
        "Semisopochnoi": {
            "marker": "P",
            "mfc": f"{colorblind_colors[2]}{alpha}",
            "mec": colorblind_colors[2],
        },
        "Ugashik": {
            "marker": "D",
            "mfc": f"{colorblind_colors[1]}{alpha}",
            "mec": colorblind_colors[1],
        },
        "Veniaminof": {
            "marker": "s",
            "mfc":f"{colorblind_colors[4]}{alpha}",
            "mec": colorblind_colors[4],

        },

    }
    return kwarg_dict