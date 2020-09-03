# unsupervised-clustering-segmentation-customers
- **Rolled up transaction level data to cutomer level, created product level data with PCA and thresholds.**
- **Segmented customers into alternative groupings with KMeans, compared alternatives using silhouette score and inertia.**
- **Analyzed feature patterns across segments, built customer profiles.**

Pakages used: **Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn**


## Project Objective
A UK based **online gift retailer** would like to segment its international customers in order to understand them better, provide more tailored services, and develop targeted marketing strategies. Currently, the segmentation is based on country, which is insufficient because:

    1. too many countries results in too many segments, and
    2. it fails to capture important factors that affect customer behaviors.

Our objective for this project is to **segment their international customers into three sub groups using transaction level data**. Additionally, we provide **insights on what the segments really mean**.

## Project Specifics
- **Delieverable**: 3 customer segments
- **Machine Learning Task**: clustering
- **Target Variable**: N/A
- **Win Condition**: N/A

## Data
1. Raw data:
    1. transaction level data over one year period, size **35,116 * 8**: line item from each invoice
    2. features include: customer ID, country, invoice number, product number, unit price, and quantity.
1. Roll up to customer level data:
     1. features include: number of orders, number of products, number of unique products, mean unit price, max unit price, total sales, mean sales per order, maximum sales on one order, minimum sales on one order.
1. One hot encode to get item level data:
     1. features are number of products purchased by each customer, there are close to 2600 features total;
     2. Number of features are reduced by either keeping only the top 20 most purchased items, or transforming to a new 127 dimension via PCA, preserving 80% of variance.

## Findings
1. Observations
    1. There are 414 customers from 36 countries. 25% of the customers are from Germany.
    2. Between 2010-12-01 to 2011-12-09, a total of 1,536 orders that include 2,574 products were made from international customers. Most items cost under £10, but there are a handful of manuals sold at a maximum of £4,000 each. Most customers purchase each item in quantities of less than 10, while some customers buy in bulk at 2,000 units per item.
    3. On average, customers put in 4 orders over the year, purchased 81 different products, generated £2,262 in sale, or £424 per order.
2. Post clustering
     1. Clusters based on customer level data only (customer clusters) and clusters based on both customer level and PCA item level data (PCA clusters) are more similar (**adjusted rand score 0.76**), compared to the similarity between customer clusters and clusters based on both customer level and top20 item level data (threshold clusters) (0.62), or the similarty between PCA clusters and threshold clusters(0.56).
     2. Customer clusters have the best silhouette score (**0.61**), to my surprise. Although, PCA clusters (0.57) or threshold item data(0.60) are only slightly lower. All three clustering seem to be of decent quality. Since the size of customer clusters is more appropriate for practical purpose (there is different in size, but no segment is too small), we choose that as our final segments.

## Segments & Insights
### Segment 0: "Occasional small shoppers"
1. This is the biggest segment. 80% of the customers belong to this segment, accounting for 37% of total sales.
2. Customers in this segment shop occasionally, averageing 2.5 orders a year and 15 products per order, generate £466 in sales, and have an average annual spending of just over £1,000.
3. They shop the least frequently, buy the least amount of products each time, and as a result, generate the least sales per order as well as total sales per customer.


### Segment 1: "Big spender or boutiques"
1. Only 17 or 4% of customers fall in this segment, yet, they accounted for 20% of total sales.
2. Customers in this segment have the highest mean order sales. Out of the 17 customers who have a mean order sales over £2,000, 16 are in this segment. 16 out of the TOP 17 most expensive orders also come from this segment.
3. Customers in this segment shop around 4 times a year, purchase 30 products per order. But their appetite for higher ticketed items push up their average sales per order to over £2,000. As a result, their annual spending is the highest among all segments at over £11,000. Almost doubling that of segment 2, and 10 times that of segment 0.
4. We can describte this segment as big boutique shoppers due to their high sales per order which is a combination of more prouducts per order and expensive items.


### Segment 2: "loyal shoppers"
1. 62 or 15.7% of customers belong to this segment, they are responsible for 42% of total sales.
2. The most distinctive feature for customers in this segment is that they all purchasd at least 100 products through out the year. This is largly due to their high shopping frequency.
3. They shop on average 10 times a year, the highest of all segments. As a result, they purchased 260 products annually compared to 130 for segment 1 and 44 for segment 0.
4. Their annual spending is over £6,000 dollars thanks to their high purchase frquency.
5. We can view this segment as customers who are loyal, as they frequently come back to order more products.

### How to use this infromation?
With the above insights, our client may choose to tailor their product offering and service to each segment. Some ideas include:
1. Most efforts should be allocated to **loyal shoppers**. Volume and frequency promotions, frequent marketing contacts, and most importantly, providing VIP customer support to make sure customers understand their loyalty is appreciated. In terms of marketing, products they frequently purchase or related products should be the most relevant.

2. For **the boutique shoppers**, individual attention should be paid. Understand their need for the higher ticketed items. Provide highly customized customer service. It is important to follow up after a purchase on an expensive item to ensure the cutomer's ultimae satisfaction. Proactivaly find lead and source for the more expensive products, individually reach out with the offerings.

3. For the **occasional small shoppers**, they can be viewed as potential other two types. Besides providing incentives of referrals, and discounts for volume/frequency, customer service should focus on their overall shopping experience, rather than on a specific product. Further product recommendations can be more diverse, less targeted.

Of course, this is just an overlook based on our analysis above. Our client will have more in house data on each customer if they dwell deeper, which will provide further insights as to how they can serve their customers better.

## Improvements
Since this is unsupervised learning, there is no natural direction we would optimize towards. However, anything could be improved. In our scenario, we could present the segments to our client and see whether our segments make sense to them. Their **feedback** can give us ideas of how to improve next.

## File Structure
The project is structured in the following way:
- **dev**: analysis notebooks
    - p1-data wrangling - transaction level
    - p2-data wrangling - customer level
    - p3-data wrangling - item level: **PCA**
    - p4-Model Delivery - make segments: **KMeans**
- **delieverables**: segment file for our client

*project rendered in nbviewer [here](https://nbviewer.jupyter.org/github/MaxineXiaoyueMa/data-science-portfolio/tree/master/clustering-segmentation-retailCustomer/dev/)*

*Disclaimer: the project scenario and data is from EliteDataScience's Machine Learning Master Class. Analysis is based on the curriculum with expansions of my own.*

**Thank you for stopping by, feel free to reach out with anything you would like to share!**
