/* Q.1 Execute the following queries in your WORKBENCH */
create database ORG;
use ORG;
CREATE TABLE Customers (
CustomerID INT PRIMARY KEY,
Name VARCHAR(255),
Email VARCHAR(255),
JoinDate DATE
);
CREATE TABLE Products (
ProductID INT PRIMARY KEY,
Name VARCHAR(255),
Category VARCHAR(255),
Price DECIMAL(10, 2)
);
CREATE TABLE Orders (
OrderID INT PRIMARY KEY,
CustomerID INT,
OrderDate DATE,
TotalAmount DECIMAL(10, 2),
FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);
CREATE TABLE OrderDetails (
OrderDetailID INT PRIMARY KEY,
OrderID INT,
ProductID INT,
Quantity INT,
PricePerUnit DECIMAL(10, 2),
FOREIGN KEY (OrderID) REFERENCES Orders(OrderID),
FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
);

INSERT INTO Customers (CustomerID, Name, Email, JoinDate) VALUES
(1, 'John Doe', 'johndoe@example.com', '2020-01-10'),
(2, 'Jane Smith', 'janesmith@example.com', '2020-01-15'),
(3, 'Akiko Tanaka', 'Akikotanaka@japan.com','1999-02-12'),
(4, 'Rafael Silva', 'Rafaelsilva@spane.com', '1998-05-15'),
(5, 'Aisha Rahaman', 'Aisharahman@arab.com', '2000-09-13'),
(6, 'Luca Rossi', 'Lucarossi@italy.com', '2001-01-14'),
(7, 'Mei Chen', 'Meichen@china.com', '1990-05-06'),
(8, 'Isabella Ferrari', 'Isabellaferrari@maxico', '1997-08-09'),
(9, 'Katrina Ivanova', 'Katrinaivanova@rassian.com', '1995-04-15'),
(10, 'Alice Johnson', 'alicejohnson@example.com', '2020-03-05');
select * from Customers;

INSERT INTO Products (ProductID, Name, Category, Price) VALUES
(1, 'Laptop', 'Electronics', 999.99),
(2, 'Smartphone', 'Electronics', 499.99),
(3, 'Fan', 'Electronics', 398.22),
(4, 'Chair', 'Furniture', 199.32),
(5, 'Bulb', 'Lighting', 177.12),
(6, 'Gas', 'Cooking', 588.34),
(7, 'Bag', 'Luggage', 433.44),
(8, 'Bottle', 'Hous hold', 44.12),
(9, 'Shirt', 'Clothes', 325.43),
(10, 'Desk Lamp', 'Home Decor', 29.99);
select * from Products;
INSERT INTO Orders (OrderID, CustomerID, OrderDate, TotalAmount)
VALUES
(1, 1, '2023-12-15', 1499.98),
(2, 2, '2023-12-17', 499.99),
(3, 3, '1998-01-12', 233.23),
(4, 4, '2001-05-13', 422.88),
(5, 5, '1999-06-17', 433.55),
(6, 6, '2005-08-19', 577.87),
(7, 7, '1992-10-19', 566.77),
(8, 8, '2008-09-16', 655.22),
(9, 9, '1995-07-21', 856.45),
(10, 10, '2020-03-21', 178.99);
select * from Orders;

INSERT INTO OrderDetails (OrderDetailID, OrderID, ProductID, Quantity,
PricePerUnit) VALUES
(1, 1, 1, 1, 999.99),
(2, 2, 2, 2, 499.99),
(3, 3, 3, 3, 977.92),
(4, 4, 4, 4, 466.77),
(5, 5, 5, 5, 29.99),
(6, 6, 6, 6, 239.96),
(7, 7, 7, 7, 765.55),
(8, 8, 8, 8, 293.88),
(9, 9, 9, 9, 288.66),
(10, 10, 10, 10, 544.98);
select * from OrderDetails;

-- 1. Basic Queries:--
-- 1.1  List all customers.--
-- Ans --
select Name from Customers;

-- 1.2 Show all products in the 'Electronics' category.--
-- Ans --
select * from Products where Category = 'Electronics';

-- 1.3. Find the total number of orders placed.--
-- Ans --
SELECT COUNT(OrderID) AS total_orders, SUM(OrderDetailID) AS total_sales
FROM orderdetails;

-- 1.4. Display the details of the most recent order.--
-- Ans--
SELECT *
FROM Orders
ORDER BY OrderDate desc
LIMIT 1;

-- 2. Joins and Relationships.--
-- 2.1. List all products along with the names of the customers who ordered them.--
-- Ans--
SELECT
    p.productid,
    p.name,
    c.name
FROM
    products p
JOIN
    orderdetails od ON p.productid = od.productid
JOIN
    orders o ON od.orderid = o.orderid
JOIN
    customers c ON o.customerid = c.customerid;
    
-- 2.2. Show orders that include more than one product.--
-- Ans --    
    
SELECT orders.orderid, COUNT(orderdetails.productid) AS product_count
FROM orders
JOIN orderdetails ON orders.orderid = orderdetails.orderid
GROUP BY orders.orderid;

-- 2.3. Find the total sales amount for each customer.--
-- Ans --
SELECT customers.customerid, customers.name, SUM(orders.TotalAmount) AS total_sales
FROM customers
JOIN orders ON customers.customerid = orders.customerid
GROUP BY customers.customerid, customers.name;

-- 3. Aggregation and Grouping --
-- 3.1. Calculate the total revenue generated by each product category.--
-- Ans --
SELECT p.category, p.name,
       SUM(od.PricePerUnit * od.quantity) AS total_revenue
FROM products p
JOIN orderdetails od ON p.productid = od.productid
JOIN orders o ON od.orderid = o.orderid
JOIN customers cust ON o.customerid = cust.customerid
GROUP BY p.category, p.name;

-- 3.2. Determine the average order value.--
-- Ans --
SELECT AVG(orders.TotalAmount) AS average_order_value
FROM orders;

-- 3.3. Find the month with the highest number of orders.--
-- Ans --
SELECT EXTRACT(MONTH FROM orderdate) AS order_month,
       COUNT(orderid) AS order_count
FROM orders
GROUP BY order_month
ORDER BY order_count DESC
LIMIT 1;

-- 4. Subqueries and Nested Queries --
-- 4.1. Identify customers who have not placed any orders.--
-- Ans --
SELECT customers.customerid, customers.name
FROM customers
LEFT JOIN orders ON customers.customerid = orders.customerid
WHERE orders.customerid IS NULL;

-- 4.2. Find products that have never been ordered. --
-- Ans --
SELECT products.productid, products.name
FROM products
LEFT JOIN orderdetails ON products.productid = orderdetails.productid
WHERE orderdetails.productid IS NULL;

-- 4.3. Show the top 3 best-selling products.--
-- Ans --
SELECT p.productid, p.name, 
       SUM(od.quantity) AS total_quantity_sold
FROM products p
JOIN orderdetails od ON p.productid = od.productid
JOIN orders o ON od.orderid = o.orderid
GROUP BY p.productid, p.name
ORDER BY total_quantity_sold DESC
LIMIT 3;

-- 5. Date and Time Functions --
-- 5.1. List orders placed in the last month--
-- Ans --
SELECT *
FROM orders
WHERE orderdate >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
AND orderdate < CURDATE();

-- 5.2. Determine the oldest customer in terms of membership duration --
-- Ans --
SELECT customerid, name, joindate
FROM customers
ORDER BY joindate ASC
LIMIT 1;

-- 6. Advanced Queries --
-- 6.1. Rank customers based on their total spending --
-- Ans --
select customers.customerid, customers.name as customername, sum(totalamount) as totalspending from orders
join customers on orders.customerid = customers.customerid group by customerid, customername
order by totalspending desc;

-- 6.2. Identify the most popular product category.--
-- Ans --
select products.Category, count(orders.OrderID) as ordercount from products
join orderdetails on products.ProductID = orderdetails.ProductID
join orders on orderdetails.OrderID = orders.OrderID
group by products.Category order by ordercount desc limit 1;					

-- 6.3. Calculate the month-over-month growth rate in sales.--
-- Ans --
select month(orderdate) as month, sum(totalamount) as growth from orders group by month order by month;

-- 7. Data Manipulation and Updates--
-- 7.1. Add a new customer to the Customers table.--
-- Ans --
INSERT INTO customers (CustomerID, Name, Email, JoinDate)
VALUES (11, 'Kane Williamson', 'KaneWilliamson@example.com', '2021-09-11');

-- 7.2. Update the price of a specific product.--
-- Ans--
UPDATE products
SET price = 19.99
WHERE productid = 1;