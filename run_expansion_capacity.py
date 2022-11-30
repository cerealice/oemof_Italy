# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:33:55 2022

@author: Alice Di Bella e Federico Canti
mail: alicedb7@gmail.com


Data
----
scenario_Italy.xlsx

Installation requirements
-------------------------
This example requires the latest version of oemof (v0.4.x) and others.
Install by:

    pip install 'oemof.solph>=0.4,<0.5'
    pip3 install xlrd
    pip3 install matplotlib
    pip3 install networkx

If you want to plot the energy system's graph, you have to install pygraphviz
using:

    pip3 install pygraphviz

For pygraphviz under Windows, some hints are available in the oemof Wiki:
https://github.com/oemof/oemof/wiki/Windows---general

5.1.2018 - uwe.krien@rl-institut.de
7.5.2018 - jonathan.amme@rl-institut.de
"""

__copyright__ = "oemof developer group"
__license__ = "GPLv3"

import os
import logging
import pandas as pd
from oemof.tools import logger
from oemof import solph
from oemof.solph import constraints
from oemof.solph import processing
from oemof.tools import economics
import time

# OEMOF function for the creation of the energy system

def nodes_from_excel(filename):
    """Read node data from Excel sheet

    Parameters
    ----------
    filename : :obj:`str`
        Path to excel file

    Returns
    -------
    :obj:`dict`
        Imported nodes data
    """

    # does Excel file exist?
    if not filename or not os.path.isfile(filename):
        raise FileNotFoundError(
            "Excel data file {} not found.".format(filename)
        )

    xls = pd.ExcelFile(filename, engine="openpyxl")

    nodes_data = {
        "buses": xls.parse("buses"),
        "commodity_sources": xls.parse("commodity_sources"),
        "transformers": xls.parse("transformers_pp"),
        "renewables": xls.parse("renewables"),
        "demand": xls.parse("demand"),
        "storages": xls.parse("storages"),
        "powerlines": xls.parse("powerlines"),
        "combined": xls.parse("combined"),
        "timeseries": xls.parse("time_series")

    }
   

    # Set datetime index -> to reduce the timeframe add [start:end] after nodes_data["timeseries"] in next line
    # Example: nodes_data["timeseries"][0:744].set_index("timestamp", inplace=True) for a single month (January) run
    nodes_data["timeseries"].set_index("timestamp", inplace=True)
    nodes_data["timeseries"].index = pd.to_datetime(
        nodes_data["timeseries"].index
    )

    print("Data from Excel file {} imported.".format(filename))

    return nodes_data


def create_nodes(nd=None):
    """Create nodes (oemof objects) from node dict

    Parameters
    ----------
    nd : :obj:`dict`
        Nodes data

    Returns
    -------
    nodes : `obj`:dict of :class:`nodes <oemof.network.Node>`
    """

    if not nd:
        raise ValueError("No nodes data provided.")

    nodes = []

    # Create Bus objects from buses table
    busd = {}

    for i, b in nd["buses"].iterrows():
        if b["active"]:
            bus = solph.Bus(label=b["label"])
            nodes.append(bus)

            busd[b["label"]] = bus
            if b["excess"]:
                nodes.append(
                    solph.Sink(
                        label=b["label"] + "_excess",
                        inputs={
                            busd[b["label"]]: solph.Flow(
                                variable_costs=b["excess costs"]
                            )
                        },
                    )
                )
            if b["shortage"]:
                nodes.append(
                    solph.Source(
                        label=b["label"] + "_shortage",
                        outputs={
                            busd[b["label"]]: solph.Flow(
                                variable_costs=b["shortage costs"]
                            )
                        },
                    )
                )

    # Create Source objects from table 'commodity sources'
    for i, cs in nd["commodity_sources"].iterrows():
        if cs["active"]:
                    
            nodes.append(
                solph.Source(
                    label=cs["label"],
                    outputs={
                        busd[cs["to"]]: solph.Flow(                                
                            variable_costs=cs["variable costs"],
                            emission_factor=cs["emission factors"],
                            # nominal_value = cs['nominal_value'], # this is if you want to put a constraint on a resource
                            # summed_max = cs['summed_max']
                        )
                    },
                )
            )
        
    # Saving data from oemof inputs
    columns = ['variable costs','emission factors']
    commodity_list = nd["commodity_sources"]["label"]
    commodity_info = pd.DataFrame(index = commodity_list, columns = columns )
    
    for i in range(len(commodity_list)):

        commodity_info.loc[commodity_list[i],'variable costs'] = float(nd["commodity_sources"]["variable costs"][i])
        commodity_info.loc[commodity_list[i],'emission factors'] = float(nd["commodity_sources"]["emission factors"][i])
        
    commodity_info.to_excel('commodity_info_table.xlsx')

    
    # Create Sink objects with fixed time series from 'demand' table
    for i, de in nd["demand"].iterrows():
        if de["active"]:
            # set static inflow values
            inflow_args = {"nominal_value": de["nominal value"]}
            # get time series for node and parameter
            for col in nd["timeseries"].columns.values:
                if col.split(".")[0] == de["label"]:
                    inflow_args[col.split(".")[1]] = nd["timeseries"][col]

            # create
            nodes.append(
                solph.Sink(
                    label=de["label"],
                    inputs={busd[de["from"]]: solph.Flow(**inflow_args)},
                )
            )

    # Saving data from OEMOF inputs that can be useful for next analysis
    
    columns = ['existing2019','added','installed','maxpotential', 'annual energy','CAPEX','OPEX','lifetime','h_eq','LCOE','LCOE_added', 'ep_cost']
    renewable_list = nd["renewables"]["label"]
    renewable_info = pd.DataFrame(index = renewable_list, columns = columns )
    
    wacc = 0.04    # value assumed for all the technologies
    
    for i in range(len(renewable_list)):

        renewable_info.loc[renewable_list[i],'existing2019'] = float(nd["renewables"]["capacity"][i])
        renewable_info.loc[renewable_list[i],'maxpotential'] = float(nd["renewables"]["maximum"][i])
        renewable_info.loc[renewable_list[i],'CAPEX'] = float(nd["renewables"]["capex"][i])
        renewable_info.loc[renewable_list[i],'OPEX'] = float(nd["renewables"]["variable_cost"][i])
        renewable_info.loc[renewable_list[i],'lifetime'] = float(nd["renewables"]["lifetime"][i])
        renewable_info.loc[renewable_list[i],'ep_cost'] = float(economics.annuity(renewable_info.loc[renewable_list[i],'CAPEX'], renewable_info.loc[renewable_list[i],'lifetime'], wacc))
        

        
        
        for col in nd["timeseries"].columns.values:
            if col.split('.')[0] == renewable_list[i]:
               renewable_info.loc[renewable_list[i],'h_eq'] = nd['timeseries'][col].sum()
               # Equivalent hours need to be modified if you run less than a year
               # Example: renewable_info.loc[renewable_list[i],'h_eq'] = nd['timeseries'][col][0:744].sum() for january

    renewable_info.to_excel('renewable_info_table.xlsx')

    # Create Source objects with fixed time series from 'renewables' table
    for i, re in nd["renewables"].iterrows():
        if re["active"]:
            
            # Values for the expansion capacity optimization
            
            capex = re['capex']
            lifetime = re['lifetime']
            wacc = wacc
            # set static outflow values
            outflow_args = {"nominal_value": re["capacity"]}
            # get time series for node and parameter
            for col in nd["timeseries"].columns.values:
                if col.split(".")[0] == re["label"]:
                    outflow_args[col.split(".")[1]] = nd["timeseries"][col]
            nodes.append(
                        solph.Source(
                            label=re["label"],
                            outputs={busd[re["to"]]: solph.Flow(fix=outflow_args['fix'],#[re["label"].split(".")[0]],  # col.split(".")[1] == ['fix]!!
                                            #fixed = True,
                                            variable_costs = re["variable_cost"],
                                            investment=solph.Investment(
                                            maximum = re['maximum'], 
                                            ep_costs = economics.annuity(capex, lifetime, wacc),
                                            existing = re['capacity']
                                            )               
                                        )
                                     },
                                )
                            )
            

    # Create Transformer objects from 'transformers' table
    for i, t in nd["transformers"].iterrows():
        if t["active"]:
            # set static inflow values
            capex = t['capex']
            lifetime = t['lifetime']
            wacc = wacc
            inflow_args = {
                "variable_costs": t["variable input costs"]
                }
            # get time series for inflow of transformer
            for col in nd["timeseries"].columns.values:
                if col.split(".")[0] == t["label"]:
                    inflow_args[col.split(".")[1]] = nd["timeseries"][col]
                    
            # get time series for outflow of transformer
            for col in nd["timeseries"].columns.values:
                if col.split(".")[0] == t["label"]:
                    outflow_args[col.split(".")[1]] = nd["timeseries"][col]

            # create
            nodes.append(
                solph.Transformer(
                    label=t["label"],
                    inputs={busd[t["from"]]: solph.Flow(**inflow_args)},
                    outputs={
                        busd[t["to"]]: solph.Flow(#max = outflow_args['fix'],
                                                  investment=solph.Investment(
                                                  maximum = t['maximum'], 
                                                  ep_costs = economics.annuity(capex, lifetime, wacc),
                                                  existing = t["capacity"]
                                            )
                            )},
                    conversion_factors={busd[t["to"]]: t["efficiency"]},
                )
            )
            
    for i, c in nd["combined"].iterrows():
        if c["active"]:
            # set static inflow values
            capex = c['capex']
            lifetime = c['lifetime']
            wacc = wacc
            inflow_args = {
                "variable_costs": c["variable input costs"]
                }
            # get time series for inflow of transformer
            for col in nd["timeseries"].columns.values:
                if col.split(".")[0] == c["label"]:
                    inflow_args[col.split(".")[1]] = nd["timeseries"][col]
            # create
            nodes.append(
                solph.Transformer(
                    label=c["label"],
                    inputs={busd[c["from"]]: solph.Flow(**inflow_args
                                                        ), busd[c["from2"]]: solph.Flow(**inflow_args)
                            
                            },
                    outputs={
                        busd[c["to"]]: solph.Flow(investment=solph.Investment(
                                            maximum = c['maximum'], 
                                            ep_costs = economics.annuity(capex, lifetime, wacc),
                                            existing = c["capacity"]
                                            )
                            )},
                    conversion_factors={busd[c["to"]]: c["efficiency"],
                                        busd[c["from"]]: c['share_bus1'], busd[c["from2"]]: c['share_bus2']
                                        }
                )
            )   
            

    transformer_list = nd['transformers']['label']
    transformer_info = pd.DataFrame(index = transformer_list, columns = columns)
    #efficiency = []

    for i in range(len(transformer_list)):  
        
        transformer_info.loc[transformer_list[i],'existing2019'] = float(nd['transformers']["capacity"][i])
        transformer_info.loc[transformer_list[i],'maxpotential'] = float(nd['transformers']["maximum"][i])
        transformer_info.loc[transformer_list[i],'efficiency'] = float(nd['transformers']["efficiency"][i])
        transformer_info.loc[transformer_list[i],'CAPEX'] = float(nd['transformers']["capex"][i])
        transformer_info.loc[transformer_list[i],'OPEX'] = float(nd['transformers']["variable input costs"][i])
        transformer_info.loc[transformer_list[i],'lifetime'] = float(nd['transformers']["lifetime"][i])
        transformer_info.loc[transformer_list[i],'ep_cost'] = float(economics.annuity(transformer_info.loc[transformer_list[i],'CAPEX'], transformer_info.loc[transformer_list[i],'lifetime'], wacc))
        #efficiency.append(float(nd['transformers']["efficiency"][i])) 
        
    transformer_info.to_excel('transformer_info_table.xlsx')
    

    for i, s in nd["storages"].iterrows():
        if s["active"]:
            
             # Values for the expansion capacity optimization
            
            capex = s['capex']
            lifetime = s['lifetime']
            wacc = wacc
            # capex_power = s['variable output costs']
          
            nodes.append(
                solph.components.GenericStorage(
                    label=s["label"],
                    inputs={
                        busd[s["bus"]]: solph.Flow(
                            nominal_value=s["capacity inflow"],
                            variable_costs=s["variable input costs"],
                            # investment=solph.Investment(
                            #                 maximum = s['maximum'], 
                            #                 ep_costs = economics.annuity(capex_power, lifetime, wacc),
                            #                 existing = s["capacity inflow"]),
                          
                        )
                    },
                    outputs={                

                        busd[s["bus"]]: solph.Flow(
                            nominal_value=s["capacity outflow"],
                            variable_costs=s["variable output costs"],
                            # investment=solph.Investment(
                            #                 maximum = s['maximum'], 
                            #                 ep_costs = economics.annuity(capex_power, lifetime, wacc),
                            #                 existing = s["capacity outflow"]),

                        )
                    },
                    
                    loss_rate=s["capacity loss"],
                    initial_storage_level=s["initial capacity"],
                    # nominal_output_capacity_ratio = 1,
                    # nominal_input_capacity_ratio = 1,
                    max_storage_level=s["capacity max"],
                    min_storage_level=s["capacity min"],
                    inflow_conversion_factor=s["efficiency inflow"],
                    outflow_conversion_factor=s["efficiency outflow"],
                    investment=solph.Investment(
                                            maximum = s['maximum'], 
                                            ep_costs = economics.annuity(capex, lifetime, wacc),
                                            existing = s["nominal capacity"]),
                )
            )
     
            
    storage_columns = ['existing2019','added','installed','maxpotential','annual energy inflow', 'annual energy outflow','h_eq', 'initial_capacity','CAPEX','OPEX_input','OPEX_output','lifetime','num_LCOS', 'ep_cost']        
    storage_list = nd['storages']['label']
    storage_info = pd.DataFrame(index = storage_list, columns = storage_columns)

    for i in range(len(storage_list)):

        storage_info.loc[storage_list[i],'existing2019'] = float(nd['storages']["nominal capacity"][i])
        storage_info.loc[storage_list[i],'maxpotential'] = float(nd['storages']["maximum"][i])
        storage_info.loc[storage_list[i],'CAPEX'] = float(nd['storages']["capex"][i])
        storage_info.loc[storage_list[i],'OPEX_input'] = float(nd['storages']["variable input costs"][i])
        storage_info.loc[storage_list[i],'OPEX_output'] = float(nd['storages']["variable output costs"][i])
        storage_info.loc[storage_list[i],'lifetime'] = float(nd['storages']["lifetime"][i])
        storage_info.loc[storage_list[i],'ep_cost'] = float(economics.annuity(storage_info.loc[storage_list[i],'CAPEX'], storage_info.loc[storage_list[i],'lifetime'], wacc))
    
    storage_info.to_excel('storage_info_table.xlsx')

    for i, p in nd["powerlines"].iterrows():
        if p["active"]:
            bus1 = busd[p["bus_1"]]
            bus2 = busd[p["bus_2"]]
            nodes.append(
                solph.custom.Link(
                    label="powerline" + "_" + p["bus_1"] + "_" + p["bus_2"],
                    inputs={bus1: solph.Flow(), bus2: solph.Flow()},
                    outputs={
                        bus1: solph.Flow(#nominal_value=p["capacity_1"],
                                          investment = solph.Investment(ep_costs = p["capex"],
                                                                        existing = p["capacity_1"],
                                                                       # maximum = 0
                                                                        )
                                         ),
                        bus2: solph.Flow(#nominal_value=p["capacity_2"],
                                          investment = solph.Investment(ep_costs = p["capex"],
                                                                        existing = p["capacity_2"],
                                                                        #maximum = 0
                                                                        )
                                         ),
                        
                    },
                    conversion_factors={
                        (bus1, bus2): p["efficiency"],
                        (bus2, bus1): p["efficiency"],
                    },
                )
            )

    powerlines_index = ['R1_R2', 'R2_R1', 'R2_R3', 'R3_R2', 'R3_R4', 'R4_R3', 'R3_R5', 'R5_R3', 'R4_R7', 'R7_R4', 'R6_R7', 'R7_R6']
    powelines_columns = ['existing2019', 'efficiency', 'added', 'installed', 'bottlenecks number', 'max transmission', 'capex', 'added energy exchanged', 'economic indicator']
    powerlines_info = pd.DataFrame(index = powerlines_index, columns = powelines_columns)

    
    for i in range(len(powerlines_index)):
        for j in range(len(nd['powerlines'].label)):
            if nd['powerlines'].label[j][0:5] == powerlines_index[i]:
                
                powerlines_info.loc[powerlines_index[i+1],'existing2019'] = nd['powerlines']['capacity_1'][j]
                powerlines_info.loc[powerlines_index[i],'existing2019'] = nd['powerlines']['capacity_2'][j]
            
                powerlines_info['efficiency'] = nd['powerlines']['efficiency'][j]
                powerlines_info['capex'] = nd['powerlines']['capex'][j]
                
    
    powerlines_info.to_excel('powerlines_info_table.xlsx')
    

    return nodes



##############################################################################
############ --- EXCEL FILES GENERATION FOR POST PROCESSING --- ##############
##############################################################################

# Simulations with various global CO2 limits

# Change the index emission based on number of scenarios and personal preference for nomenclature
    
index_emission = ['-100%', '-90%','-80%','-70%','-60%','-50%','-40%','-30%','-20%','-10%', '-0%'] # for 11 iterations

# List of strings useful for the post processing of results
name_tech = ['water','RH','Geo','Bio','import','gas', 'wind','solar','solar_US','batt','phs', 'elect', 'h2', 'fc','combined']
parameters_energy = ['load', 'overgeneration', 'charge_batt', 'charge_phs', 'total_costs','gas_mix', 'h2_to_pipeline_to_electricity', 'total_gas_without_hydrogen','transmission_losses', 'commoditygas','emissions']
column_energy = name_tech + parameters_energy
columns_h2 = ['from electrolyzer','to fuel cell','h2_mix', 'gas_mix', 'gas_alone', 'h2 share','h2_tank']
    
res_tech = ['R1_wind', 'R1_solar', 'R1_solar_US', 
            'R2_wind', 'R2_solar',  'R2_solar_US',
            'R3_wind', 'R3_solar',  'R3_solar_US', 
            'R4_wind', 'R4_solar',  'R4_solar_US', 
            'R5_wind', 'R5_solar',  'R5_solar_US',
            'R6_wind', 'R6_solar',  'R6_solar_US', 
            'R7_wind', 'R7_solar',  'R7_solar_US']

capacities = [f'capacity_{i}' for i in name_tech]
name_tot = name_tech + capacities 
parameters_costs = ['total_costs', 'commoditygas','gas_price', 'powerlines']
columns_costs = name_tot + parameters_costs


# Initializing post processing dataframes
capacity_existing_2019 = pd.DataFrame(0, index = index_emission, columns = name_tech)
energy_generated = pd.DataFrame(0, index = index_emission, columns = column_energy)
capacity_added = pd.DataFrame(0,index = index_emission, columns = name_tech)  
capacity_installed = pd.DataFrame(0, index = index_emission, columns = name_tech)
costs_data = pd.DataFrame(0, index = index_emission, columns = columns_costs)
h2_info = pd.DataFrame(0, index = index_emission, columns = columns_h2)

##############################################################################
############# --- CO2 EMISSIONS CONSTRAINT AND INPUT DATA  --- ###############
##############################################################################

initial_emis = 50 # [Mton of eqCO2] very high value for the first run. You should run a single iteration with this very high constraint 
                    # just to obtain the value for emissions of the first iteration, set that as the initial_emis and run the 11 iterations
regions_number = 7 # select the region number depending on electricity market of the country
scenarios_number = 2 # select the number of scenarios. It is one for the first iteration to know the emissions value, then set it to the number you prefer
red_step = 0.1 # how much the emission reduction step is. 0.1 is for 11 iterations, for 6 iterations could be 0.2 and so on (remeber the first iteration goes with no emission constraints)
CO2_values = []
red = 0
conversion = 1e6 # for conversion
mese = 12 # Depending on how long the time span is. Please put 1 if all year round, 12 for a single month
h2_percentage = 0.2 #Specify here the share of h2 in the gas pipelines 
elect_efficiency = 0.8
safety_coefficient = 0.99 # per powelines

for perc in range(scenarios_number):
    CO2_values.append(initial_emis*(1-red))
    red += red_step

##############################################################################
######################### --- ITERATIVE RUNS  --- ############################
##############################################################################

for kk in range(len(CO2_values)):
    ca = CO2_values[kk]
    
    logger.define_logging()
    datetime_index = pd.date_range(
        "2021-01-01 00:00:00", "2021-12-31 23:00:00", freq = '60min'
    )
    
    # model creation and solving
    logging.info("Starting optimization")
    
    # initialisation of the energy system
    esys = solph.EnergySystem(timeindex=datetime_index)
    
    # read node data from Excel sheet
    data_name = "scenario_Italy"
    excel_nodes = nodes_from_excel(os.path.join(os.getcwd(), f"{data_name}.xlsx",))
    
    # create nodes from Excel sheet data
    my_nodes = create_nodes(nd=excel_nodes)
    
    # add nodes and flows to energy system
    esys.add(*my_nodes)
    
    
    # creation of a least cost model from the energy system
    
    om = solph.Model(esys)
    
    om.receive_duals()
    
    # add the emission constraint; add again ca = max Mton of CO2
    
    constraints.emission_limit(om, limit=ca*conversion)
    
    start_time = time.time()
    
    # solving the linear problem using the given solver
    om.solve(solver="gurobi", cmdline_options={"LogFile": 'LogFile',
                                                #"Method": 3,
                                                #"DualReductions":0
                                                #"Threads": 8
                                                }
              )
    
    
    middle_time = time.time()
    print("Solving took %.2f minutes" % ((middle_time - start_time)/60)) 
    
    # print out the amount of emissions from the emission constraint
    print(f"Emissions are {om.integral_limit_emission_factor()/conversion}")

    results = processing.results(om)
    
##############################################################################
################# --- DATA POST PROCESSING AND SAVING  --- ###################
##############################################################################

    
    commodity_info = pd.read_excel('commodity_info_table.xlsx', index_col = 'label', engine = "openpyxl")
    
    transformer_info = pd.read_excel('transformer_info_table.xlsx', index_col = 'label', engine = "openpyxl")
    transformer_info = transformer_info.fillna(0)
    
    energy_generated.loc[len(CO2_values)-1-kk,'total_costs'] = solph.processing.meta_results(om)['objective']/conversion
    costs_data.loc[len(CO2_values)-1-kk,'total_costs'] = solph.processing.meta_results(om)['objective']
    energy_generated.loc[len(CO2_values)-1-kk,'commoditygas'] = solph.views.node(results, 'GL_bus_gas')['sequences'].sum()[-1]
    costs_data.loc[len(CO2_values)-1-kk,'commoditygas'] += energy_generated.loc[len(CO2_values)-1-kk,'commoditygas'] * commodity_info.loc['GL_resource_gas','variable costs']
    energy_generated.loc[len(CO2_values)-1-kk,'emissions'] = om.integral_limit_emission_factor()/conversion
    energy_generated.loc[len(CO2_values)-1-kk,'gas_price'] = commodity_info.loc['GL_resource_gas','variable costs']
    
                 
    
    region = {}
    national_load = 0
    national_excess = 0
    
    for i in range(regions_number):
        region[f'R{i+1}'] = solph.views.node(results, f'R{i+1}_bus_el')['sequences']
        national_load += float(solph.views.node(results, f'R{i+1}_load')['sequences'].sum())
        national_excess += float(solph.views.node(results, f'R{i+1}_bus_el_excess')['sequences'].sum())
       
    for i in region:
        region[i].to_excel(f'Italia_regions_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx', sheet_name = i)
    
    # Retrieve data from renewable dataframe
        
    renewable_info = pd.read_excel('renewable_info_table.xlsx', index_col = 'label', engine = "openpyxl")
    
    for i in renewable_info.index:
    
        renewable_info.loc[i,'annual energy'] = solph.views.node(results, i)['sequences'].sum()[0]
        renewable_info.loc[i,'added'] = solph.views.node(results, i)['scalars'][0]
        renewable_info.loc[i,'installed'] = renewable_info.loc[i,'added'] + renewable_info.loc[i,'existing2019']
        
        if renewable_info.loc[i,'annual energy'] == 0:
            renewable_info.loc[i,'LCOE'] = float('nan')
        elif renewable_info.loc[i,'annual energy']-renewable_info.loc[i,'existing2019']*renewable_info.loc[i,'h_eq'] == 0:
            renewable_info.loc[i,'LCOE_added'] = float('nan')
        else:    
            renewable_info.loc[i,'LCOE'] = renewable_info.loc[i,'CAPEX']/renewable_info.loc[i,'lifetime']/mese/renewable_info.loc[i,'h_eq']+renewable_info.loc[i,'OPEX']
            renewable_info.loc[i,'LCOE_added'] = renewable_info.loc[i,'CAPEX']*renewable_info.loc[i,'added']/renewable_info.loc[i,'lifetime']/mese/(renewable_info.loc[i,'annual energy']-renewable_info.loc[i,'existing2019']*renewable_info.loc[i,'h_eq'])+renewable_info.loc[i,'OPEX']
    
    # Retrieve data from transformers dataframe
    
    for i in transformer_info.index:
    
        transformer_info.loc[i,'annual energy'] = solph.views.node(results, i)['sequences'].sum()[1]
        transformer_info.loc[i,'added'] = solph.views.node(results, i)['scalars'][0]
        transformer_info.loc[i,'installed'] = transformer_info.loc[i,'added'] + transformer_info.loc[i,'existing2019']
        
        if transformer_info.loc[i,'installed'] == 0:
            transformer_info.loc[i,'h_eq'] = float('nan')
        else:    
            transformer_info.loc[i,'h_eq'] = transformer_info.loc[i,'annual energy']/transformer_info.loc[i,'installed']
        
        if transformer_info.loc[i,'annual energy'] == 0:
            transformer_info.loc[i,'LCOE'] = float('nan')
        else:      
            transformer_info.loc[i,'LCOE'] = transformer_info.loc[i,'CAPEX'] * transformer_info.loc[i,'added']/transformer_info.loc[i,'lifetime']/ mese / transformer_info.loc[i,'annual energy'] + transformer_info.loc[i,'OPEX']
        
        if i.split('_')[2] == 'gas':
            transformer_info.loc[i,'annual energy'] = transformer_info['annual energy'][i]*transformer_info.loc[i,'efficiency']     
    
        if i.split('_')[2] == 'elect':
            transformer_info.loc[i,'annual energy'] = solph.views.node(results, i)['sequences'].sum()[0]
            
            
    # Retrieve data from storage dataframe
    
    storage_info = pd.read_excel('storage_info_table.xlsx', index_col = 'label', engine = "openpyxl")
    
    for i in storage_info.index:
    
        storage_info.loc[i,'annual energy inflow'] = solph.views.node(results, i)['sequences'].sum()[0]
        storage_info.loc[i,'annual energy outflow'] = solph.views.node(results, i)['sequences'].sum()[2]
        storage_info.loc[i,'added'] = solph.views.node(results, i)['scalars'][1]        
        storage_info.loc[i,'installed'] = storage_info.loc[i,'added'] + storage_info.loc[i,'existing2019']
        storage_info.loc[i,'num_LCOS'] = storage_info.loc[i,'installed'] * storage_info.loc[i,'CAPEX']/storage_info.loc[i,'lifetime']
    
    
    
    index = energy_generated.index
    
    for i in energy_generated.columns:
        
        for j in renewable_info.index: 
            if j.split('_')[-1] == i.split('_')[-1]:
                energy_generated.loc[len(CO2_values)-1-kk, i]  = renewable_info.loc[j,'annual energy']
                costs_data.loc[len(CO2_values)-1-kk,i]  = renewable_info.loc[j,'annual energy'] * renewable_info.loc[j,'OPEX']
                costs_data.loc[len(CO2_values)-1-kk, f'capacity_{i}']  = renewable_info.loc[j,'added'] * renewable_info.loc[j,'ep_cost']
                capacity_added.loc[len(CO2_values)-1-kk, i]  = renewable_info.loc[j,'added']
                capacity_installed.loc[len(CO2_values)-1-kk, i]  = renewable_info.loc[j,'installed']
                capacity_existing_2019.loc[len(CO2_values)-1-kk, i]  = renewable_info.loc[j,'existing2019']
                
        for j in transformer_info.index:
            if j.split('_')[-1] == i.split('_')[-1]:
                energy_generated.loc[len(CO2_values)-1-kk, i] = transformer_info.loc[j,'annual energy']
                costs_data.loc[len(CO2_values)-1-kk, i]  = transformer_info.loc[j,'annual energy'] * transformer_info.loc[j,'OPEX']           
                costs_data.loc[len(CO2_values)-1-kk, f'capacity_{i}'] = transformer_info.loc[j,'added'] * transformer_info.loc[j,'ep_cost']                
                capacity_added.loc[len(CO2_values)-1-kk, i]  = transformer_info.loc[j,'added']
                capacity_installed.loc[len(CO2_values)-1-kk, i]  = transformer_info.loc[j,'installed']
                capacity_existing_2019.loc[len(CO2_values)-1-kk, i]  = transformer_info.loc[j,'existing2019']
            
    
        for j in storage_info.index:
            if j.split('_')[-1] == i.split('_')[-1]:
                            
                if i.split('_')[0] == 'charge':
                    
                    energy_generated.loc[len(CO2_values)-1-kk, i]  = storage_info.loc[j,'annual energy inflow']
                    
                else:
                    
                    energy_generated.loc[len(CO2_values)-1-kk, i]  = storage_info.loc[j,'annual energy outflow']
                    costs_data.loc[len(CO2_values)-1-kk, f'capacity_{i}']  = storage_info.loc[j,'added'] * storage_info.loc[j,'ep_cost']               
    
                    
                    capacity_existing_2019.loc[len(CO2_values)-1-kk, i]  = storage_info.loc[j,'existing2019']
                    capacity_added.loc[len(CO2_values)-1-kk, i]  = storage_info.loc[j,'added']
                    capacity_installed.loc[len(CO2_values)-1-kk, i]  = storage_info.loc[j,'installed']
                    
    
    energy_generated.loc[len(CO2_values)-1-kk, 'load']  = national_load
    energy_generated.loc[len(CO2_values)-1-kk, 'overgeneration']  = national_excess
    energy_generated.loc[len(CO2_values)-1-kk, 'gas_mix'] = energy_generated.loc[len(CO2_values)-1-kk, 'combined']-energy_generated.loc[len(CO2_values)-1-kk, 'gas']
    energy_generated.loc[len(CO2_values)-1-kk, 'h2_to_pipeline_to_electricity'] = energy_generated.loc[len(CO2_values)-1-kk, 'gas_mix']*h2_percentage
    energy_generated.loc[len(CO2_values)-1-kk, 'total_gas_without_hydrogen'] = energy_generated.loc[len(CO2_values)-1-kk, 'combined']-energy_generated.loc[len(CO2_values)-1-kk, 'h2_to_pipeline_to_electricity']

    costs_data.loc[len(CO2_values)-1-kk, 'commoditygas'] += energy_generated.loc[len(CO2_values)-1-kk, 'total_gas_without_hydrogen'] / transformer_info.loc['R1_pp_gas', 'efficiency'] * commodity_info.loc['GL_resource_gas', 'variable costs']              
            
    
    ##########################################################################
    ###################### --- DECISION VARIABLES DATA --- ###################
    ##########################################################################
    
    index_regions = ['R1', 'R2','R3','R4','R5','R6', 'R7']
    variable_renewable_info = renewable_info
    variable_renewable_info['regional_production'] = 0
    variable_renewable_info['overgeneration'] = 0
    regional_generation_sum = pd.DataFrame(index = index_regions)
    regional_generation_sum['production'] = 0
    
    for i in variable_renewable_info.index:
        condition = i.split('_')[-1] 
        if condition == 'Geo' or condition == 'Bio' or condition == 'RH':
            variable_renewable_info = variable_renewable_info.drop(i)
    
    
    for i in range(len(variable_renewable_info.index)):
        
        for j in range(regions_number):
            
            if variable_renewable_info.index[i].split('_')[0] == f'R{j+1}':
                regional_generation_sum.loc[regional_generation_sum.index[j],'production'] += variable_renewable_info.loc[variable_renewable_info.index[i],'annual energy']
            #MODIFICATO FINO A QUI, VERIFICARE PRIMA CHE GLI INDICI NUMERICI VENGONO LETTI CORRETTAMENTE
        
            
    for i in range(len(variable_renewable_info.index)):
        for j in range(regions_number):
            if variable_renewable_info.index[i].split('_')[0] == f'R{j+1}':
                variable_renewable_info.loc[variable_renewable_info.index[i],'regional_production'] = regional_generation_sum.iloc[j][0]
                
    for i in range(len(variable_renewable_info.index)):
                    
        for j in index_regions:
            
            if variable_renewable_info.index[i].split('_')[0] == j:
                variable_renewable_info.loc[variable_renewable_info.index[i],'overgeneration'] = solph.views.node(results, f'{j}_bus_el_excess')['sequences'].sum()[0]
                
            
    variable_renewable_info['h_eq_fittizie'] = (variable_renewable_info['annual energy'] - variable_renewable_info['overgeneration'] * variable_renewable_info['annual energy'] / variable_renewable_info['regional_production']) / variable_renewable_info['installed']
    variable_renewable_info['LCOE_fittizio'] = variable_renewable_info['CAPEX'] / variable_renewable_info['lifetime'] / mese / variable_renewable_info['h_eq_fittizie'] + variable_renewable_info['OPEX']
    variable_renewable_info['LCOE_finale'] = variable_renewable_info['ep_cost'] / variable_renewable_info['h_eq'] + variable_renewable_info['OPEX'] 
    storage_info['LCOS'] = storage_info['CAPEX'] * storage_info['installed'] / storage_info['lifetime'] / mese / storage_info['annual energy outflow'] + storage_info['OPEX_output']
    
    for i in storage_info.index:
        
        for j in transformer_info.index:
            
            if i.split('_')[2] == 'h2':
                    if i.split('_')[0] == j.split('_')[0] and j.split('_')[2] == 'elect':
    
                        storage_info.loc[i,'num_LCOS'] += transformer_info.loc[j,'CAPEX']*transformer_info.loc[j,'installed'] / transformer_info.loc[j,'lifetime']
                
                    if i.split('_')[0] == j.split('_')[0] and j.split('_')[2] == 'fc':
    
                        storage_info.loc[i,'num_LCOS'] += transformer_info.loc[j,'CAPEX']*transformer_info.loc[j,'installed'] / transformer_info.loc[j,'lifetime']
                        
        
    
    ##########################################################################
    ###################### -- POWERLINES  -- #################################
    ##########################################################################
    
    powerlines_info = pd.read_excel('powerlines_info_table.xlsx', engine = "openpyxl")
    powerlines_info = powerlines_info.set_index(powerlines_info['Unnamed: 0'])
    powerlines_info = powerlines_info.drop(['Unnamed: 0'], axis=1)
    powerlines_info['added energy exchanged'] = 0
        
    powerlines = ['R1_R2', 'R1_R2', 'R2_R3', 'R2_R3', 'R3_R4', 'R3_R4', 'R3_R5', 'R3_R5', 'R4_R7', 'R4_R7', 'R6_R7', 'R6_R7']
        
    timestep_pow = pd.DataFrame(index = solph.views.node(results, 'R1_pp_water')['sequences'].index, columns = powerlines_info.index)
    
    added_pow = []
    
    powerlines_transmission_losses = pd.DataFrame(index=powerlines_info.index, columns = ('losses', 'op'))
    
    for i in range(len(powerlines_info.index)):
        
            first = powerlines[i].split('_')[0]
            second = powerlines[i].split('_')[1]   
            
            if first == powerlines_info.index[i].split('_')[0]:
                added_pow.append(solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['scalars'][1])
                timestep_pow[powerlines_info.index[i]] = solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['sequences'][solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['sequences'].columns[3]]
                
                powerlines_transmission_losses.loc[powerlines_info.index[i],'losses'] = solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el')['sequences'].sum()[0] - solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el')['sequences'].sum()[3]
                
                
            if second == powerlines_info.index[i].split('_')[0]:
                added_pow.append(solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['scalars'][0])
                timestep_pow[powerlines_info.index[i]] = solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['sequences'][solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el', f'{first}_bus_el')['sequences'].columns[2]]
                
                powerlines_transmission_losses.loc[powerlines_info.index[i],'losses'] = solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el')['sequences'].sum()[1] - solph.views.node(results, f'powerline_{first}_bus_el_{second}_bus_el')['sequences'].sum()[2]
    
    powerlines_transmission_losses = powerlines_transmission_losses.drop('op', axis=1)
    energy_generated.loc[len(CO2_values)-1-kk,'transmission_losses']  = powerlines_transmission_losses.sum()[0]
    
    
    for i in powerlines_info.index:
        powerlines_info.loc[i,'max transmission'] = timestep_pow[i].max()
        powerlines_info.loc[i,'existing2019'] = powerlines_info.loc[i,'existing2019']
        powerlines_info.loc[i,'added'] =  added_pow[powerlines_info.index.get_loc(i)]
        powerlines_info.loc[i,'installed'] = powerlines_info.loc[i,'existing2019'] + powerlines_info.loc[i,'added']
        powerlines_info.loc[i,'bottlenecks number'] = sum(1 for j in timestep_pow[i] if j > powerlines_info.loc[i,'installed']*safety_coefficient)
    
    for j in range(len(powerlines_info.index)):
        for i in range(len(timestep_pow)):
            if timestep_pow.iloc[i][j] > powerlines_info['existing2019'][j]:
                powerlines_info.loc[powerlines_info.index[j],'added energy exchanged'] += timestep_pow.iloc[i][j] - powerlines_info.loc[powerlines_info.index[j],'existing2019']
    
    powerlines_info['economic indicator'] = powerlines_info['capex'] * powerlines_info['added'] / powerlines_info['added energy exchanged']
    costs_data['powerlines']  = powerlines_info['added'].sum() * powerlines_info['capex'][0]
    
    
    sum_h2 = 0
    electro = 0
    sum_h2_mix = 0
    sum_gl_mix = 0
    only_gas = 0
    h2_tank = 0
    
    for i in range(regions_number):
        
        electro += solph.views.node(results, f'R{i+1}_pp_elect')['sequences'].sum()[1]
        sum_h2 += solph.views.node(results, f'R{i+1}_pp_fc')['sequences'].sum()[0]
        sum_h2_mix += solph.views.node(results, f'R{i+1}_pp_mix')['sequences'].sum()[1]
        sum_gl_mix += solph.views.node(results, f'R{i+1}_pp_mix')['sequences'].sum()[0]
        only_gas += solph.views.node(results, f'R{i+1}_bus_gas')['sequences'].sum()[2]
            
        
        for j in storage_info.index:
            if j.split('_')[2] == 'h2':
                storage_info.loc[j,'LCOS'] = storage_info.loc[j,'num_LCOS']/(solph.views.node(results, f'R{i+1}_pp_gas')['sequences'].sum()[1]*commodity_info.loc['GL_resource_gas', 'variable costs'] + solph.views.node(results, f'R{i+1}_pp_fc')['sequences'].sum()[1])
    
        
    for j in storage_info.index:
        
        if j.split('_')[-1] == 'h2':
            h2_tank += storage_info.loc[j, 'installed']
            
    
    h2_info.loc[len(CO2_values)-1-kk, 'to fuel cell']  = sum_h2
    h2_info.loc[len(CO2_values)-1-kk, 'from electrolyzer']  = electro   
    h2_info.loc[len(CO2_values)-1-kk, 'h2_mix']= sum_h2_mix
    h2_info.loc[len(CO2_values)-1-kk, 'gas_mix'] = sum_gl_mix
    h2_info.loc[len(CO2_values)-1-kk, 'gas_alone'] = only_gas
    h2_info.loc[len(CO2_values)-1-kk, 'h2 share'] = h2_info.loc[len(CO2_values)-1-kk, 'h2_mix']/h2_info.loc[len(CO2_values)-1-kk, 'gas_mix']*100

    h2_info.loc[len(CO2_values)-1-kk, 'h2_tank']  = h2_tank
    
    storage_content_timeseries = pd.DataFrame()
    
    for i in range(regions_number):
    
        storage_content_timeseries[f'R{i+1}_h2'] = solph.views.node(results, f'R{i+1}_storage_h2')['sequences'].iloc[:,1]
        storage_content_timeseries[f'R{i+1}_batt'] = solph.views.node(results, f'R{i+1}_storage_batt')['sequences'].iloc[:,1]
        storage_content_timeseries[f'R{i+1}_phs'] = solph.views.node(results, f'R{i+1}_storage_phs')['sequences'].iloc[:,1]


    ##########################################################################
    ######################### -- SAVE DATA  -- ###############################
    ##########################################################################
    
    renewable_info.to_excel(f'renewable_info_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    transformer_info.to_excel(f'transformer_info_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    variable_renewable_info.to_excel(f'variable_renewable_info_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    storage_info.to_excel(f'storage_info_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    powerlines_info.to_excel(f'powerlines_info_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    timestep_pow.to_excel(f'timestep_powerlines_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
    
    storage_content_timeseries.to_excel(f'storage_content_timeseries_scenario_[{index_emission[len(index_emission)-1-kk]}].xlsx')
            
    energy_generated.to_excel(f'energy_generated_[{index_emission[len(index_emission)-1-kk]}].xlsx')

    costs_data.to_excel(f'costs_data_[{index_emission[len(index_emission)-1-kk]}].xlsx')
        
    capacity_added.to_excel(f'capacity_added_[{index_emission[len(index_emission)-1-kk]}].xlsx')
        
    capacity_existing_2019.to_excel(f'capacity_existing_2019_[{index_emission[len(index_emission)-1-kk]}].xlsx')

    h2_info.to_excel(f'h2_info_[{index_emission[len(index_emission)-1-kk]}].xlsx')
        

logging.info("Done!")

