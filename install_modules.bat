# PowerShell script to create and configure a conda environment on WINDOWS

# Default environment name
$default_env_name = "lszpy_env"

# Prompt user
$env_name = Read-Host "Enter the name of the environment to create (press Enter to use '$default_env_name')"
if ([string]::IsNullOrWhiteSpace($env_name)) {
    $env_name = $default_env_name
}

# Check if environment exists
$env_exists = conda env list | Select-String -Pattern "^\s*$env_name\s"
if ($env_exists) {
    Write-Host "Environment '$env_name' already exists. Exiting script."
    exit 1
}

# Create the environment
Write-Host "> Creating conda env with python=3.10 from conda-forge"
conda create -n $env_name python=3.10 -c conda-forge -y

# Activate the environment
Write-Host "> Activating conda env"
conda activate $env_name

# Set channel priority to strict
Write-Host "> Setting channel priority to strict"
conda config --env --set channel_priority strict

# Install scientific core packages
Write-Host "> Installing scientific core packages"
conda install numpy scipy matplotlib shapely netCDF4 gdal pyproj -c conda-forge -y --repodata-fn=repodata.json

Write-Host "> Installing Harmonica module"
pip install harmonica

# Install plotting tools
Write-Host "> Installing interactive plotting tools"
conda install ipympl ipywidgets -c conda-forge -y

# Install logging / PDF tools
Write-Host "> Installing logging / PDF tools"
pip install pdfkit
conda install -c conda-forge wkhtmltopdf -y
