# sort method_dirs by Z_noise, then by W_noise within that
def sort_by_noise_level(file_list):
    """ Expects filenames to be of the form Z_noise:3-W_noise:1
    where the order of Z_noise and W_noise can change, and their
    respective values can change, but they must be separated by a hyphen
    """

    # split on hyphen key.split("-") and sort to ensure Z noise comes first, then W
    convert = lambda filename: sorted(filename.split("-"), reverse=True)

    # drop textual characters, leaving only numeric noise levels (Z's then W's noise level) to use for sorting
    my_key = lambda key: [float(''.join(i for i in s if i.isdigit() or i == '.')) for s in convert(key)]
    return sorted(file_list, key=my_key)
