import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
import numpy as np
import pickle
import sklearn
from datetime import date
import time

# Setting up page configuration

st.set_page_config(page_title= "Industrial copper modeling",
                   layout= "wide",
                   initial_sidebar_state= "expanded"                   
                  )

# Creating option menu in the side bar

with st.sidebar:

    selected = option_menu("Menu", ["Home","Selling Price","Status"], 
                           icons=["house","cash","award"],
                           menu_icon= "menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "orange"},
                                   "nav-link-selected": {"background-color": "orange"}}
                          )
    
if selected == 'Home':

    st.title(":orange[*INDUSTRIAL COPPER MODELING*]")
    
    col1, col2 = st.columns(2)
    with col1:
        col1.markdown("# ")
        col1.markdown("## :blue[*Overview*] : Build regression model to predict selling price and classification model to predict status")
        col1.markdown("# ")
        col1.markdown("## :blue[*Domain*] : Copper Manufacturing")
        col1.markdown("# ")
        col1.markdown("## :blue[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit.")
        

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/images-5.jpeg")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/images-4.jpeg")

if selected == 'Selling Price':
    col1, col2, col3 = st.columns([4, 10, 2])
    with col2:
        st.title(":orange[*PREDICT SELLING PRICE*]")
    col1, col2, col3 = st.columns([4, 10, 7])
    with col2:
        colored_header(
            label="",
            description="",
            color_name="blue-green-70"
        )
    col1, col2, col3 = st.columns([2, 10, 2])
    # Start from options
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")

    with col2:

        # Quantity Ton

        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Quantity  </span><span style='color: violet;'> Ton </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=0.1,max_value=1000000000.0")
        qt = st.number_input('', min_value=0.1, max_value=1000000000.0, value=1.0)
        quantity_log = np.log(qt)

        # Customer Value

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Customer  </span><span style='color: violet;'> Value </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=12458.0,max_value=2147483647.0")
        customer = st.number_input('', min_value=12458.0, max_value=2147483647.0, value=12458.0, )

        # Country Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Country  </span><span style='color: violet;'> Code </span> </h1>",
            unsafe_allow_html=True)
        country = st.selectbox(' ', [28, 38, 78, 27, 30, 32, 77, 25, 113, 26, 39, 40, 84, 80, 79, 89, 107])

        # Item Type

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Item  </span><span style='color: violet;'> Type </span> </h1>",
            unsafe_allow_html=True)
        cc = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
        item_type = st.selectbox('          ', cc)

        # Application Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Application </span><span style='color: violet;'> Code </span> </h1>",
            unsafe_allow_html=True)
        av = st.selectbox('          ', [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0,
                                         27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0,
                                         59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0])

        application_log = np.log(av)

        # Product Referal Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Product </span><span style='color: violet;'> Referal Code</span> </h1>",
            unsafe_allow_html=True)

        pr = [1670798778, 611993, 1668701376, 164141591, 628377,
              1671863738, 640665, 1332077137, 1668701718, 640405,
              1693867550, 1665572374, 1282007633, 1668701698, 628117,
              1690738206, 640400, 1671876026, 628112, 164336407,
              164337175, 1668701725, 1665572032, 611728, 1721130331,
              1693867563, 611733, 1690738219, 1722207579, 1665584662,
              1665584642, 929423819, 1665584320]
        product_ref = st.selectbox("", pr)

        # Thickness Value

        with col2:
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Thickness  </span><span style='color: violet;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=0.1, max_value=2500.000000")
            thickness = st.number_input('', min_value=0.1, max_value=2500.000000, value=1.0)
            thickness_log = np.log(thickness)

            # Width Value

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Width  </span><span style='color: violet;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=1.0, max_value=2990.000000")
            wv = st.number_input('', min_value=1.0, max_value=2990.000000, value=1.0)
            width_log = np.log(wv)

            # Item Date

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Item  </span><span style='color: violet;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(1995,1,1),max_Date(2021,12,31)")
            item_date = st.date_input(label='', min_value=date(1995, 1, 1),
                                      max_value=date(2021, 12, 31), value=date(2021, 8, 1))
            
            # Delivery Date

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Delivery </span><span style='color: violet;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(2020,1,1),max_date=date(2023,12,31)")
            delivery_date = st.date_input(label='    ', min_value=date(2020, 1, 1),
                                          max_value=date(2023, 12, 31), value=date(2021, 8, 1))
            
            # Status Code

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Status </span><span style='color: violet;'> Code </span> </h1>",
                unsafe_allow_html=True)
            status_code = {'Won': 1, 'Draft': 2, 'To be approved': 3, 'Lost': 0, 'Not lost for AM': 5, 'Wonderful': 6,
                           'Revised': 7, 'Offered': 8, 'Offerable': 4}
            Status = st.selectbox('             ', status_code)

            # To Predict Selling Price

            predict_data = [quantity_log, customer, country, cc[item_type], application_log, thickness_log, width_log,
                            product_ref, item_date.day,
                            item_date.month, item_date.year, delivery_date.day, delivery_date.month, delivery_date.year,
                            status_code[Status]]
            #print(predict_data)
            with open('regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
               # print(type(model))
        col1, col2, col3 = st.columns([10, 1, 10])

        with col1:
            st.write("")
            if st.button('Process'):
                x = model.predict([predict_data])
               # print(x)
                st.markdown(
                    f"<h1 style='font-size: 40px;'><span style='color: orange;'>Predicted Selling Price : </span><span style='color: green;'> {np.exp(x[0])}</span> </h1>",
                    unsafe_allow_html=True)

# Predict Status
        
if selected == 'Status':
    col1, col2, col3 = st.columns([4, 10, 2])
    with col2:
        st.title(":orange[*PREDICT STATUS*]")
    col1, col2, col3 = st.columns([4, 10, 5])
    with col2:
        colored_header(
            label="",
            description="",
            color_name="blue-green-70"
        )
    col1, col2, col3 = st.columns([2, 10, 2])
    # Start from options
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")

    with col2:

        # Quantity Value

        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Quantity  </span><span style='color: violet;'> Ton </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=0.1,max_value=1000000000.0")
        qt = st.number_input('', min_value=0.1, max_value=1000000000.0, value=1.0)
        quantity_log = np.log(qt)

        # Customer Value

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Customer  </span><span style='color: violet;'> Value </span> </h1>",
            unsafe_allow_html=True)
        st.markdown("Note: min_value=12458.0,max_value=2147483647.0")
        customer = st.number_input('', min_value=12458.0, max_value=2147483647.0, value=12458.0, )

        # Country Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Country  </span><span style='color: violet;'> Code </span> </h1>",
            unsafe_allow_html=True)
        country = st.selectbox(' ', [28, 38, 78, 27, 30, 32, 77, 25, 113, 26, 39, 40, 84, 80, 79, 89, 107])

        # Item Type
        
        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Item  </span><span style='color: violet;'> Type </span> </h1>",
            unsafe_allow_html=True)
        cc = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
        item_type = st.selectbox('          ', cc)

        # Application Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Application </span><span style='color: violet;'> Code </span> </h1>",
            unsafe_allow_html=True)
        av = st.selectbox('          ', [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0,
                                         27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0,
                                         59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0])

        application_log = np.log(av)

        # Product Referal Code

        st.write("")
        st.write("")
        st.markdown(
            "<h1 style='font-size: 40px;'><span style='color: violet;'>Product </span><span style='color: violet;'> Referal Code</span> </h1>",
            unsafe_allow_html=True)

        pr = [1670798778, 611993, 1668701376, 164141591, 628377,
              1671863738, 640665, 1332077137, 1668701718, 640405,
              1693867550, 1665572374, 1282007633, 1668701698, 628117,
              1690738206, 640400, 1671876026, 628112, 164336407,
              164337175, 1668701725, 1665572032, 611728, 1721130331,
              1693867563, 611733, 1690738219, 1722207579, 1665584662,
              1665584642, 929423819, 1665584320]
        product_ref = st.selectbox("", pr)

        # Thickness Value

        with col2:
            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Thickness  </span><span style='color: violet;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=0.1, max_value=2500.000000")
            thickness = st.number_input('', min_value=0.1, max_value=2500.000000, value=1.0)
            thickness_log = np.log(thickness)

            # Width Value

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Width  </span><span style='color: violet;'> Value </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=1.0, max_value=2990.000000")
            wv = st.number_input('', min_value=1.0, max_value=2990.000000, value=1.0)
            width_log = np.log(wv)

            # Item Date

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Item  </span><span style='color: violet;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(1995,1,1),max_Date(2021,12,31)")
            item_date = st.date_input(label='', min_value=date(1995, 1, 1),
                                      max_value=date(2021, 12, 31), value=date(2021, 8, 1))
            
            # Delivery Date

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Delivery </span><span style='color: violet;'> Date </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_date(2020,1,1),max_date=date(2023,12,31)")
            delivery_date = st.date_input(label='    ', min_value=date(2020, 1, 1),
                                          max_value=date(2023, 12, 31), value=date(2021, 8, 1))
            
            # Selling Price

            st.write("")
            st.write("")
            st.markdown(
                "<h1 style='font-size: 40px;'><span style='color: violet;'>Selling </span><span style='color: violet;'> Price </span> </h1>",
                unsafe_allow_html=True)
            st.markdown("Note: min_value=1.0, max_value=100001015.0")
            sp = st.number_input('', min_value=1.0, max_value=100001015.0, value=1.0)
            selling_price = np.log(sp)

            predict_data = [quantity_log, customer, country, cc[item_type], application_log, thickness_log, width_log,
                            product_ref, item_date.day,
                            item_date.month, item_date.year, delivery_date.day, delivery_date.month, delivery_date.year,
                            selling_price]

            with open('classification_dataset.pkl', 'rb') as f:
                model = pickle.load(f)
        col1, col2, col3 = st.columns([10, 2, 10])

        with col1:
            st.write("")
            if st.button('Process'):
                x = model.predict([predict_data])
                if x[0] == 1:
                   # print(x)
                    st.markdown(
                        "<h1 style='font-size: 40px;'><span style='color: orange;'>Predicted Status : </span><span style='color: green;'> Won </span> </h1>",
                        unsafe_allow_html=True)

                elif x[0] == 0:
                    #print(x)
                    st.markdown(
                        "<h1 style='font-size: 40px;'><span style='color: orange;'>Predicted Status : </span><span style='color: red;'> Lost </span> </h1>",
                        unsafe_allow_html=True)
                    
