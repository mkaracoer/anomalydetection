import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
import rasterio.plot
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.ndimage import convolve
from skimage.transform import probabilistic_hough_line


matplotlib.use('TkAgg')
np.random.seed(42)


# calculate the possible train lying angle range
def get_train_angle_range(array, delta):
    non_nan_indices = np.where(~np.isnan(array))

    x_min = np.min(non_nan_indices[1])
    x_max = np.max(non_nan_indices[1])

    x_min_y_min = np.min(np.where(~np.isnan(array[:, x_min])))
    x_min_y_max = np.max(np.where(~np.isnan(array[:, x_min])))

    x_max_y_min = np.min(np.where(~np.isnan(array[:, x_max])))
    x_max_y_max = np.max(np.where(~np.isnan(array[:, x_max])))

    coord00 = (x_min, -x_min_y_min)
    coord01 = (x_max, -x_max_y_max)

    coord10 = (x_min, -x_min_y_max)
    coord11 = (x_max, -x_max_y_min)

    angle = []

    dx = coord01[0] - coord00[0]
    dy = coord01[1] - coord00[1]
    angle.append(math.atan2(dy, dx))

    dx = coord11[0] - coord10[0]
    dy = coord11[1] - coord10[1]
    angle.append(math.atan2(dy, dx))

    angle = sorted(angle)

    # add -5 degrees +5 degrees
    angle[0] = angle[0] + np.radians(-delta)
    angle[1] = angle[1] + np.radians(+delta)
    return np.array(angle)


# function for normalization to 0-255
def normalize(array, pol=1, threshold_min=0.1, threshold_max=0.2):
    array[np.isnan(array)] = 0
    normalized_array = np.where(array < threshold_min, 0, array)
    normalized_array = np.where(normalized_array > threshold_max, threshold_max,
                                normalized_array)
    normalized_array = 255 * ((normalized_array - normalized_array.min()) /
                              normalized_array.max()) ** pol
    return normalized_array.astype(int)


# adding buffers around a True cell
def add_buffer_zone(grid, buffer_size):
    kernel = np.ones((2 * buffer_size + 1, 2 * buffer_size + 1), dtype=bool)
    convolved_grid = convolve(grid, kernel, mode='constant', cval=False)
    result_grid = convolved_grid > 0
    return result_grid


# detecting cells where the value in 3 arrays are bigger than a value
def detect_cells(arr1, arr2, arr3):
    mask1 = add_buffer_zone(arr1 >= 200, 1)  # arr1 >= 200
    mask2 = add_buffer_zone(arr2 >= 200, 1)  # arr2 >= 200
    mask3 = add_buffer_zone(arr3 >= 200, 1)  # arr3 >= 200
    detection = np.all(np.array([mask1, mask2, mask3]), axis=0)
    detection = add_buffer_zone(detection, 1)
    return detection


def geo_to_pixel(geometry, transform):
    """Transform a Shapely geometry from geographic to pixel coordinates."""
    # Apply the inverse transform to each coordinate
    def transform_point(x, y):
        row, col = ~transform * (x, y)
        return int(round(row)), int(round(col))

    if isinstance(geometry, Point):
        # For Point geometries
        x, y = geometry.x, geometry.y
        return Point(transform_point(x, y))
    elif isinstance(geometry, Polygon):
        # For Polygon geometries
        transformed_exterior = [transform_point(x, y) for x, y in geometry.exterior.coords]
        transformed_interiors = [
            [transform_point(x, y) for x, y in interior.coords]
            for interior in geometry.interiors
        ]
        return Polygon(transformed_exterior, transformed_interiors)
    else:
        raise TypeError("Geometry type not supported")


# plotting function
def plot(z, name, trains=None, osm_buildings=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    # ax.axis('off')
    ax.imshow(z, interpolation='none')

    if trains is not None:
        for train in trains:
            p0, p1 = train
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')

            # draw angle in degrees over the line
            # slope = (p1[1]-p0[1]) / (p1[0]-p0[0])
            # angle = np.math.atan(slope)
            # ax.annotate(text=f'{round(np.degrees(angle))}', xy=(p0[0], p0[1]),
            #             c='red')

        # angle range drawing
        # ax.plot((0, 218), (0, 135))
        # ax.plot((0, 218), (3, 133))
        # slope = (135 - 0) / (218 - 0)
        # angle = np.math.atan(slope)
        # ax.annotate(text=f'{round(np.degrees(angle))}', xy=(0, 0),
        #             c='red')

    if osm_buildings is not None:
        osm_buildings.plot(ax=ax, color='red')

    plt.savefig(f'plot/{name}.svg', bbox_inches='tight')
    plt.close()
    return None


if __name__ == '__main__':
    # parameter selection ######################################################
    name = 'muenchen_2023'
    deepness = 10
    pol, min_, max_ = 1, 0.1, 0.2

    # image read ###############################################################
    img = rasterio.open(f'thesis_data/{name}.tif')
    out_image = img.read()
    osm_buildings = gpd.read_file(f'osm_buildings/{name[:-5]}.shp')\
        .to_crs(img.crs)

    # transform ################################################################
    trans = img.transform
    osm_buildings['geometry'] = osm_buildings['geometry'].apply(
        lambda geom: geo_to_pixel(geom, trans))

    # calculate possible train angle ###########################################
    angle = get_train_angle_range(out_image[0], delta=4)
    print('Rad', angle)
    print('Deg', np.degrees(angle))

    # building detection #######################################################
    all_masks = []
    for _ in range(deepness):
        r1 = np.random.randint(0, len(out_image))
        r2 = np.random.randint(0, len(out_image))
        r3 = np.random.randint(0, len(out_image))
        final_mask = detect_cells(
            normalize(out_image[r1], pol=pol, threshold_min=min_,
                      threshold_max=max_),
            normalize(out_image[r2], pol=pol, threshold_min=min_,
                      threshold_max=max_),
            normalize(out_image[r3], pol=pol, threshold_min=min_,
                      threshold_max=max_))
        all_masks.append(final_mask)

    building = np.any(np.array(all_masks), axis=0)  # combine union
    # building = np.all(np.array(all_masks), axis=0)  # combine intersection

    dates = []
    for i in range(len(out_image)):
        # normalized bands #####################################################
        norm_img = normalize(out_image[i], pol=pol, threshold_min=min_,
                             threshold_max=max_)

        # plotting #############################################################
        # plot generation without building detection
        z = np.dstack((norm_img, norm_img, norm_img))
        plot(z, f'{name}_{img.descriptions[i]}_without_building_mask',
             osm_buildings=osm_buildings)

        # plot generation with building detection
        norm_img[building] = 0
        z = np.dstack((norm_img, norm_img, norm_img))
        plot(z, f'{name}_{img.descriptions[i]}_with_building_mask')

        # line detection #######################################################
        trains = probabilistic_hough_line(norm_img, threshold=5, line_length=5,
                                          line_gap=3, theta=np.pi/2-angle,
                                          seed=42)
        z = np.dstack((norm_img, norm_img, norm_img))
        plot(z, f'{name}_{img.descriptions[i]}_with_train_detection', trains)

        print(f'{img.descriptions[i]}: At least {len(trains)} '
              f'Train(s) is/are detected!')

        if len(trains) > 0:
            dates.append(img.descriptions[i])

    print(dates)
