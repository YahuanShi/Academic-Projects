import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import warnings 
warnings.filterwarnings("ignore")

# Create Dash app
app = dash.Dash(__name__)

# Read CSV file
df_gesamt = pd.read_csv('final_data_group_20.csv', sep=None, engine='python', encoding='ISO-8859-1')

# Vehicle type data
anzahl_typ = df_gesamt['Type_Fahrzeug'].value_counts()

# Data processing
OEM1 = df_gesamt[df_gesamt['Herstellernummer_Fahrzeug'] == 1]
OEM2 = df_gesamt[df_gesamt['Herstellernummer_Fahrzeug'] == 2]

# Month processing
OEM1.loc[:, 'Monat_Fahrzeug'] = pd.to_datetime(OEM1['Produktionsdatum_Fahrzeug']).dt.month
OEM2.loc[:, 'Monat_Fahrzeug'] = pd.to_datetime(OEM2['Produktionsdatum_Fahrzeug']).dt.month
anzahl_monat1 = OEM1['Monat_Fahrzeug'].value_counts()
anzahl_monat2 = OEM2['Monat_Fahrzeug'].value_counts()

# Year processing
OEM1.loc[:, 'Year_Fahrzeug'] = pd.to_datetime(OEM1['Produktionsdatum_Fahrzeug']).dt.year
OEM2.loc[:, 'Year_Fahrzeug'] = pd.to_datetime(OEM2['Produktionsdatum_Fahrzeug']).dt.year
anzahl_year1 = OEM1['Year_Fahrzeug'].value_counts()
anzahl_year2 = OEM2['Year_Fahrzeug'].value_counts()

# OEM1 2012-2014 data
year2012_Typ1=OEM1[OEM1['Year_Fahrzeug'] == 2012]['Type_Fahrzeug'].value_counts()
year2013_Typ1=OEM1[OEM1['Year_Fahrzeug'] == 2013]['Type_Fahrzeug'].value_counts()
year2014_Typ1=OEM1[OEM1['Year_Fahrzeug'] == 2014]['Type_Fahrzeug'].value_counts()

OEM1_year_typ = pd.DataFrame({
    'Jahr': [2012, 2013, 2014],
    'number type 11': [year2012_Typ1[11], year2013_Typ1[11], year2014_Typ1[11]],
    'number type 12': [year2012_Typ1[12], year2013_Typ1[12], year2014_Typ1[12]]
})

# OEM2 2012-2014 data
year2012_Typ2=OEM2[OEM2['Year_Fahrzeug'] == 2012]['Type_Fahrzeug'].value_counts()
year2013_Typ2=OEM2[OEM2['Year_Fahrzeug'] == 2013]['Type_Fahrzeug'].value_counts()
year2014_Typ2=OEM2[OEM2['Year_Fahrzeug'] == 2014]['Type_Fahrzeug'].value_counts()

OEM2_year_typ = pd.DataFrame({
    'Jahr': [2012, 2013, 2014],
    'number type 21': [year2012_Typ2[21], year2013_Typ2[21], year2014_Typ2[21]],
    'number type 22': [year2012_Typ2[22], year2013_Typ2[22], year2014_Typ2[22]]
})

# Production location data
anzahl_Ort1 = OEM1['ORT_Fahrzeug'].value_counts()
anzahl_Ort2 = OEM2['ORT_Fahrzeug'].value_counts()

# Vehicle type per location
anzahl_Nuernberg = df_gesamt[df_gesamt['ORT_Fahrzeug'] == 'NUERNBERG']['Type_Fahrzeug'].value_counts()
anzahl_Bonn = df_gesamt[df_gesamt['ORT_Fahrzeug'] == 'BONN']['Type_Fahrzeug'].value_counts()
anzahl_Goettingen = df_gesamt[df_gesamt['ORT_Fahrzeug'] == 'GOETTINGEN']['Type_Fahrzeug'].value_counts()
anzahl_Regensburg = df_gesamt[df_gesamt['ORT_Fahrzeug'] == 'REGENSBURG']['Type_Fahrzeug'].value_counts()

# print(anzahl_Nuernberg, anzahl_Bonn, anzahl_Goettingen, anzahl_Regensburg)
# OEM1 only in BONN and NUERNBERG
# OEM2 only in GOETTINGEN, REGENSBURG

OEM1_Ort_typ = pd.DataFrame({
    'Ort': anzahl_Ort1.index,
    'number type 11': [anzahl_Bonn.get(11, 0), anzahl_Nuernberg.get(11, 0)],  
    'number type 12': [anzahl_Bonn.get(12, 0), anzahl_Nuernberg.get(12, 0)]   
})

# Creating OEM2_Ort_typ DataFrame
OEM2_Ort_typ = pd.DataFrame({
    'Ort': anzahl_Ort2.index,
    'number type 21': [anzahl_Goettingen.get(21, 0), anzahl_Regensburg.get(21, 0)],  
    'number type 22': [anzahl_Goettingen.get(22, 0), anzahl_Regensburg.get(22, 0)]   
})

# Customer data
customer = pd.read_csv('Data/Zulassungen/Zulassungen_alle_Fahrzeuge.csv', 
                        delimiter=';', 
                        quotechar='"', 
                        usecols=["IDNummer", "Gemeinden", "Zulassung"], on_bad_lines='skip')
customer['Zulassung'] = pd.to_datetime(customer['Zulassung'])
customer_filtered = customer[customer['Zulassung'] >= '2012-01-01']
merged_table = pd.merge(df_gesamt, 
                         customer_filtered[['IDNummer', 'Gemeinden', 'Zulassung']], 
                         left_on='ID_Fahrzeug', 
                         right_on='IDNummer', 
                         how='left')
merged_table.drop(columns=["IDNummer"], inplace=True)

# Top registration locations for OEM1 and OEM2
anzahl_Zulassung_Ort1 = merged_table[merged_table['Herstellernummer_Fahrzeug'] == 1]['Gemeinden_y'].value_counts()
anzahl_Zulassung_Ort2 = merged_table[merged_table['Herstellernummer_Fahrzeug'] == 2]['Gemeinden_y'].value_counts()
N = 10
top_N_OEM1 = anzahl_Zulassung_Ort1.nlargest(N)
top_N_OEM2 = anzahl_Zulassung_Ort2.nlargest(N)

# Logo and color setup
logo_url = "https://upload.wikimedia.org/wikipedia/en/9/95/Technical_University_of_Berlin_logo.png"
light_blue_color = '#ADD8E6'

# Create chart functions
def create_vehicle_type_bar_chart():
    valid_anzahl_typ = anzahl_typ[anzahl_typ > 0]  
    # desired_types = [11, 12, 21, 22]
    # filtered_anzahl_typ = valid_anzahl_typ[valid_anzahl_typ.index.isin(desired_types)]

    fig = px.bar(
        x=valid_anzahl_typ.index.astype(str), 
        y=valid_anzahl_typ.values,  
        labels={'x': 'Vehicle Type', 'y': 'Number'},
        title="Which vehicle types have parts with premature rust spots?"
    )
    
    return fig

def create_production_month_bar_chart(anzahl_monat1, anzahl_monat2):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=anzahl_monat1.index, y=anzahl_monat1.values, name='OEM1'))
    fig.add_trace(go.Bar(x=anzahl_monat2.index, y=anzahl_monat2.values, name='OEM2'))
    fig.update_layout(barmode='group', title="Production month vehicles by OEM1 and OEM2",
                      xaxis_title="Month", yaxis_title="Number")
    return fig

def create_production_year_bar_chart(anzahl_year1, anzahl_year2):
    valid_anzahl_year1 = anzahl_year1[anzahl_year1 > 0]
    valid_anzahl_year2 = anzahl_year2[anzahl_year2 > 0]

    common_years = valid_anzahl_year1.index.intersection(valid_anzahl_year2.index)

    valid_anzahl_year1 = valid_anzahl_year1[valid_anzahl_year1.index.isin(common_years)]
    valid_anzahl_year2 = valid_anzahl_year2[valid_anzahl_year2.index.isin(common_years)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=valid_anzahl_year1.index, y=valid_anzahl_year1.values, name='OEM1'))
    fig.add_trace(go.Bar(x=valid_anzahl_year2.index, y=valid_anzahl_year2.values, name='OEM2'))
    fig.update_layout(barmode='group', title="Production year vehicles by OEM1 and OEM2",
                      xaxis_title="Year", yaxis_title="Number")
    return fig

def create_production_location_pie_chart(anzahl_Ort1, anzahl_Ort2):
    fig1 = go.Figure(data=[go.Pie(labels=anzahl_Ort1.index, values=anzahl_Ort1.values, hole=.3)])
    fig2 = go.Figure(data=[go.Pie(labels=anzahl_Ort2.index, values=anzahl_Ort2.values, hole=.3)])
    
    fig1.update_layout(title_text="Production location of vehicles with premature rust spots by OEM1")
    fig2.update_layout(title_text="Production location of vehicles with premature rust spots by OEM2")
    
    return fig1, fig2

def create_year_type_bar_chart_OEM1():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=OEM1_year_typ['Jahr'], y=OEM1_year_typ['number type 11'], name='number type 11'))
    fig.add_trace(go.Bar(x=OEM1_year_typ['Jahr'], y=OEM1_year_typ['number type 12'], name='number type 12'))
    fig.update_layout(barmode='group', title="Production year and type by OEM1", xaxis_title="Year", yaxis_title="Number")
    return fig

def create_year_type_bar_chart_OEM2():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=OEM2_year_typ['Jahr'], y=OEM2_year_typ['number type 21'], name='number type 21'))
    fig.add_trace(go.Bar(x=OEM2_year_typ['Jahr'], y=OEM2_year_typ['number type 22'], name='number type 22'))
    fig.update_layout(barmode='group', title="Production year and type by OEM2", xaxis_title="Year", yaxis_title="Number")
    return fig

def create_customer_location_chart(top_N_OEM1, top_N_OEM2):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_N_OEM1.index, y=top_N_OEM1.values, name='OEM1', marker_color='blue'))
    fig.add_trace(go.Bar(x=top_N_OEM2.index, y=top_N_OEM2.values, name='OEM2', marker_color='green'))
    fig.update_layout(barmode='group', title="Top 10 car registration locations of vehicles with premature rust spots",
                      xaxis_title="Location", yaxis_title="Number of registrations")
    return fig

def create_production_location_type_OEM1():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=OEM1_Ort_typ['Ort'], y=OEM1_Ort_typ['number type 11'], name='number type 11'))
    fig.add_trace(go.Bar(x=OEM1_Ort_typ['Ort'], y=OEM1_Ort_typ['number type 12'], name='number type 12'))
    fig.update_layout(barmode='group', title="Production location and type by OEM1", xaxis_title="Location", yaxis_title="Number")
    return fig

def create_production_location_type_OEM2():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=OEM2_Ort_typ['Ort'], y=OEM2_Ort_typ['number type 21'], name='number type 21'))
    fig.add_trace(go.Bar(x=OEM2_Ort_typ['Ort'], y=OEM2_Ort_typ['number type 22'], name='number type 22'))
    fig.update_layout(barmode='group', title="Production location and type by OEM2", xaxis_title="Location", yaxis_title="Number")
    return fig

# Define layout
app.layout = html.Div(style={'backgroundColor': light_blue_color, 'font-family': 'Source Sans Pro'}, children=[
    html.Div([
        html.Img(src=logo_url, style={'height':'10%', 'width':'10%', 'float': 'right'}),
        html.H1("Vehicle Rust Data Dashboard", style={'textAlign': 'center', 'color': '#000080'})
    ]),
    
    html.Div([
        html.H2("Vehicle Type Analysis"),
        dcc.Graph(figure=create_vehicle_type_bar_chart())
    ]),

    html.Div([
        html.H2("Production Month Analysis"),
        dcc.Graph(figure=create_production_month_bar_chart(anzahl_monat1, anzahl_monat2))
    ]),

    html.Div([
        html.H2("Production Year Analysis"),
        dcc.Graph(figure=create_production_year_bar_chart(anzahl_year1, anzahl_year2))
    ]),

    html.Div([
        html.H2("Production Year and Type by OEM1"),
        dcc.Graph(figure=create_year_type_bar_chart_OEM1())
    ]),

    html.Div([
        html.H2("Production Year and Type by OEM2"),
        dcc.Graph(figure=create_year_type_bar_chart_OEM2())
    ]),
    
    html.Div([
        html.H2("Production Location Analysis (OEM1)"),
        dcc.Graph(figure=create_production_location_pie_chart(anzahl_Ort1, anzahl_Ort2)[0])
    ]),

    html.Div([
        html.H2("Production Location Analysis (OEM2)"),
        dcc.Graph(figure=create_production_location_pie_chart(anzahl_Ort1, anzahl_Ort2)[1])
    ]),

    html.Div([
        html.H2("Production location and type by OEM1"),
        dcc.Graph(figure=create_production_location_type_OEM1())
    ]),

    html.Div([
        html.H2("Production location and type by OEM2"),
        dcc.Graph(figure=create_production_location_type_OEM2())
    ]),

    html.Div([
        html.H2("Top Registration Locations"),
        dcc.Graph(figure=create_customer_location_chart(top_N_OEM1, top_N_OEM2))
    ])
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
