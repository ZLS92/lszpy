import argparse
import raster_tools

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--raster', required=True, help='Input raster file')
    parser.add_argument('--lim', nargs='+', type=float, help='Limits for the raster')
    parser.add_argument('--xRes', type=float, help='X resolution')
    parser.add_argument('--yRes', type=float, help='Y resolution')
    parser.add_argument('--width', type=int, help='Width of the output raster')
    parser.add_argument('--height', type=int, help='Height of the output raster')
    parser.add_argument('--lim_prjcode', help='Projection code for the limits')
    parser.add_argument('--out_prjcode', help='Output projection code')
    parser.add_argument('--new_path', default='/vsimem/', help='Path for the new raster')
    parser.add_argument('--new_name', help='Name for the new raster')
    parser.add_argument('--srcNodata', type=float, help='No data value for the source raster')
    parser.add_argument('--dstNodata', type=float, help='No data value for the destination raster')
    parser.add_argument('--extension', default='vrt', help='File extension for the new raster')
    parser.add_argument('--method', default='GRA_Average', help='Resampling method')
    parser.add_argument('--cutlineDSName', help='Cutline dataset name')
    parser.add_argument('--cropToCutline', type=bool, default=False, help='Crop to cutline')
    parser.add_argument('--tps', type=bool, default=False, help='Use thin plate spline transformation')
    parser.add_argument('--rpc', type=bool, default=False, help='Use rational polynomial coefficients')
    parser.add_argument('--geoloc', type=bool, default=False, help='Use geolocation array')
    parser.add_argument('--errorThreshold', type=float, default=0, help='Error threshold')
    parser.add_argument('--options', nargs='+', default=[], help='Additional options')
    parser.add_argument('--plot', type=bool, default=False, help='Plot the output raster')
    parser.add_argument('--vmin', type=float, help='Minimum value for the plot')
    parser.add_argument('--vmax', type=float, help='Maximum value for the plot')

    args = parser.parse_args()

    method_mapping = {
        'GRA_Average': raster_tools.gdal.GRA_Average,
        # add other methods here
    }

    # Get the method from the arguments
    method = method_mapping[args.method]

    raster_tools.raster_warp( raster= args.raster, 
                              lim= args.lim, 
                              xRes= args.xRes, 
                              yRes= args.yRes, 
                              width= args.width, 
                              height= args.height, 
                              lim_prjcode= args.lim_prjcode, 
                              out_prjcode= args.out_prjcode, 
                              new_path= args.new_path, 
                              new_name= args.new_name, 
                              srcNodata= args.srcNodata, 
                              dstNodata= args.dstNodata, 
                              close= True,
                              extension= args.extension, 
                              method= method,  # use 'method' instead of 'args.method'
                              cutlineDSName= args.cutlineDSName, 
                              cropToCutline= args.cropToCutline, 
                              tps= args.tps, 
                              rpc= args.rpc, 
                              geoloc= args.geoloc, 
                              errorThreshold= args.errorThreshold, 
                              options= args.options, 
                              plot= args.plot, 
                              vmin= args.vmin, 
                              vmax= args.vmax )

if __name__ == "__main__":
    main()