from openslide import OpenSlide
def get_mpp_and_resolution(slide: OpenSlide):

    res_unit = slide.properties['tiff.ResolutionUnit']
    x_res = float(slide.properties['tiff.XResolution'])
    y_res = float(slide.properties['tiff.YResolution'])
    assert x_res == y_res
    res = int(x_res)

    match res_unit:
        case 'centimeter':
            mpp = 1e4 / res
        case _:
            raise ValueError(f"Unknown resolution unit: {res_unit}")

    return mpp, res

def get_mpp(slide: OpenSlide):
    mpp_x = float(slide.properties['openslide.mpp-x'])
    mpp_y = float(slide.properties['openslide.mpp-y'])
    assert mpp_x == mpp_y

    return mpp_x


