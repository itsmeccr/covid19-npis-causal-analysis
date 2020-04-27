## Project: A Meta-Learning Approach to Causal Structure Discovery from Unknown Interventions
### University of Illinois at Chicago 
#### Advanced Machine Learning


Data preparation for application in the causal analysis of non-pharmaceutical interventions on COVID19 cases.

### Demographic Data Integration
- Population
- Population Density
- Elderly Population Fraction
- Urban/Rural
Sources:
https://www.ers.usda.gov/webdocs/DataFiles/48747/PopulationEstimates.csv?v=3011.3
https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv
https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_Gaz_counties_national.zip
https://www2.census.gov/programs-surveys/popest/tables/2010-2018/municipios/asrh/PEP_2018_PEPAGESEX.zip 

### Flights Data Integration
- Number of flights in a state last week
Source: https://opensky-network.org/datasets/covid-19/

### Climate Data Integration
- Average air temperature in a state last week
Source: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#datasets
(Avialable as Google BigQuery Public Dataset)

### Interventions Data Integration
- Non-pharmaceutical intervention policies enforced by authorities
Source: https://github.com/Keystone-Strategy/covid19-intervention-data

### COVID19 Cases Data Integration
Source: https://github.com/CSSEGISandData/COVID-19
(Avialable as Google BigQuery Public Dataset)