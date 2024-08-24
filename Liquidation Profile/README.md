# liquidation-profile

## The Risk Manager wants to analyze the asset liquidation/monetization profile of the book. The book has four assets as mentioned below.

| Asset Name | Currency | Notional         |
|------------|----------|------------------|
| Asset1     | USD      | 1,000,000.00     |
| Asset2     | USD      | 2,000,000.00     |
| Asset3     | SGD      | 3,000,000.00     |
| Asset4     | SGD      | 4,000,000.00     |

### Liquidation Schedule

Based on assets in the book, the desk has provided the liquidation start date, the number of days required to liquidate an asset, and the expected cash flow date based on the currency calendar.

| Asset Name | Liquidating Start Date | No of days liquidation | Expected Cash Flow Start date |
|------------|------------------------|------------------------|-------------------------------|
| Asset1     | 6-Aug-2024 (Business Day + 2WD) | 3 | 7-Aug-2024 (Business Day + 3WD) |
| Asset2     | 8-Aug-2024 (Business Day + 4WD) | 2 | 9-Aug-2024 (Business Day + 5WD) |
| Asset3     | 8-Aug-2024 (Business Day + 4WD) | 6 | 12-Aug-2024 (Business Day + 5WD) |
| Asset4     | 5-Aug-2024 (Business Day + 1WD) | 5 | 6-Aug-2024 (Business Day + 2WD) |

### Holiday Calendar

| Currency | Date     |
|----------|----------|
| SGD      | 3-Aug-24 |
| SGD      | 4-Aug-24 |
| SGD      | 10-Aug-24 |
| SGD      | 11-Aug-24 |
| USD      | 3-Aug-24 |
| USD      | 4-Aug-24 |
| USD      | 10-Aug-24 |
| USD      | 11-Aug-24 |

## Requirement

As a member of the risk infra team, you are requested to develop and provide a code either in SQL or Python (or both) which will help the Risk Manager to generate:

a. The expected liquidity cashflow profile based on the information provided by the desk. (Sample data provided below)

b. The solution should be extendable to N assets denominated in different currencies.

### Expected Cash Flow Profile

| Asset Name | Currency | Expected cash flow Amount | Expected Cash Flow Date |
|------------|----------|--------------------------|-------------------------|
| Asset1     | USD      | 333,333.33               | 7-Aug-24                |
| Asset1     | USD      | 333,333.33               | 8-Aug-24                |
| Asset1     | USD      | 333,333.33               | 9-Aug-24                |
| Asset2     | USD      | 1,000,000.00             | 9-Aug-24                |
| Asset2     | USD      | 1,000,000.00             | 12-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 12-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 13-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 14-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 15-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 16-Aug-24               |
| Asset3     | SGD      | 500,000.00               | 17-Aug-24               |
| Asset4     | SGD      | 800,000.00               | 6-Aug-24                |
| Asset4     | SGD      | 800,000.00               | 7-Aug-24                |
| Asset4     | SGD      | 800,000.00               | 8-Aug-24                |
| Asset4     | SGD      | 800,000.00               | 9-Aug-24                |
| Asset4     | SGD      | 800,000.00               | 12-Aug-24               |

---