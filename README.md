# IA651 Final Project

## Project Description
This is the final project for the course ia651 (Applied Machine Learning) of Clarkson University (Spring 2023). The goal for the final project is to select an appropriate private or public dataset and build a predictive model for it. Data cleaning, exploration and preparing should be done to prepare the data to build the prediction model. 

For the selected Airbnb dataset start with being familiar with dataset, cleaning dataset, exploring and explaining, finding relations and building model and doing prediction. 

## NYC Airbnb listing Dataset
Airbnb is an online marketplace that connects people who are looking for a place to stay with people who have extra space to rent out. It was founded in 2008 and has since grown into a global platform with listings in over 220 countries and regions. On Airbnb, hosts can list their homes, apartments, guesthouses, or other types of accommodations, and travelers can search for and book these spaces for short-term stays. Airbnb also offers experiences and activities hosted by local experts in various locations, allowing travelers to immerse themselves in the culture and community of their destination.

The Airbnb NYC listing dataset is a collection of information about listings in New York City that are available on the Airbnb platform. It contains information such as the location, type of accommodation, price, number of bedrooms, amenities, availability, and reviews from guests. The dataset was first released in 2015 and has been updated annually since then.

The dataset is publicly available and has been used by researchers to analyze various aspects of the Airbnb market in New York City. For example, researchers have used the dataset to study the impact of Airbnb on the housing market, the distribution of listings across neighborhoods, and the characteristics of hosts and guests. The dataset includes information on over 50,000 listings in New York City, making it one of the largest Airbnb datasets available.

## Resources
**Dataset** : [NYC Airbnb 2023](http://data.insideairbnb.com/united-states/ny/new-york-city/2023-03-06/data/listings.csv.gz/)

**Python Version** : 3.11

**Packages**: Numpy, Pandas, Keras, Matplotlib, Scikit, Seaborn

**IDE** : Visual Studio Code, Jupyter Notebook


## Dataset features

List of the important features which we will use in this project.

* **id**: A unique identifier for each Airbnb listing.
* **name**: The name or title of the Airbnb listing.
* **host_id**: A unique identifier for the host of the Airbnb listing.
* **host_name**: The name of the host of the Airbnb listing.
* **neighbourhood_group**: The borough (i.e. neighborhood group) in which the Airbnb listing is located.
* **neighbourhood**: The specific neighborhood in which the Airbnb listing is located.
* **latitude**: The latitude coordinate of the Airbnb listing's location.
* **longitude**: The longitude coordinate of the Airbnb listing's location.
* **room_type**: The type of room or space that is being rented (e.g. Entire home/apt, Private room, Shared room).
* **price**: The nightly price for the Airbnb listing.
* **minimum_nights**: The minimum number of nights that a renter must stay in the Airbnb listing.
* **number_of_reviews**: The number of reviews that the Airbnb listing has received.
* **last_review**: The date of the most recent review of the Airbnb listing.
* **reviews_per_month**: The average number of reviews per month for the Airbnb listing.
* **calculated_host_listings_count**: The number of listings that the host has on Airbnb.
* **availability_365**: The number of days that the Airbnb listing is available for rent during the next 365 days.

* **amenities**: List of amenities reported by the host.
* **bedrooms**: number of bedrooms available for use.
* **beds**: number of beds.
* **bathrooms**: number of bathrooms available for use.
* **accommodates**: number of people who can use the listing

## Data cleaning

The first step is to know the dataset itself and it's general status. 

```python
df.shape
```





