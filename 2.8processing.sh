cd "/Users/cosminbalmos/Documents/Computer Science Y4/Thesis/Inputs"

cdo griddes "1.4degree-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc" | head -10

echo "Creating 280km file using nearest neighbor remapping..."
cdo remapnn,r23x9 \
    "1.4degree-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc" \
    "280km-nobnds-nopress-fullgreenland.KNMI-1950-2099.FGRN11.BN_RACMO2.3p2_CESM2_FGRN11.DD.nc"

