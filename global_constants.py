IMAGE_SIZE = 224

SCALED_DIMENSIONS = {
    248: (112, 224),
    190: (212, 224),
    180: (224, 224),
    208: (224, 189)
}

PAD_AMOUNT = {
    248: ((56, 56), None),
    190: ((6, 6), None),
    180: (None, None),
    208: (None, (17, 18))
}