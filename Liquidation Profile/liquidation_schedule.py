import pandas as pd
import numpy as np

class liquidation:
    def __init__(
            self,
            assets: pd.DataFrame,
            holiday: pd.DataFrame,
            liquidation_schedule: pd.DataFrame
    ) -> None:
        
        self.assets: pd.DataFrame = assets
        self.holiday: pd.DataFrame = holiday
        self.liquidation: pd.DataFrame = liquidation_schedule

    def get_asset_details(
            self, 
            asset: str
        ) -> tuple[np.array, int, str, float]:
        """
        Retrieves asset details

        Parameters
        ----------
        asset: str
            Asset of interest

        Returns
        -------
        tuple[np.array, int, str, float]
            start date of expected cash flow
            Liquidation days
            Currency
            Notional
        """

        target_asset: pd.DataFrame = self.assets[self.assets['Asset Name'] == asset]
        target_liquidation: pd.DataFrame = self.liquidation[self.liquidation['Asset Name'] == asset]
        start_date: np.array = target_liquidation['Expected Cash Flow Start Date'].values[0]
        days: int = target_liquidation['No of days liquidation'].values[0]
        currency: str = target_asset['Currency'].values[0]
        notional: float = target_asset['Notional'].values[0]

        return start_date, days, currency, notional

    def generate_cash_flow_dates(
            self, 
            start_date: np.array, 
            days: int, 
            currency: str
        ) -> pd.DataFrame:
        """
        Computes expected cash flow dates

        Parameters
        ----------
        start_date: np.array
            Expected cash flow start date
        days: int
            Number of days for liquidation
        currency: str
            Currency of Asset (for tracking holidays)

        Returns
        -------
        pd.DataFrame
            DataFrame of expected cash flow dates
        """
        currency_holiday: pd.DataFrame = self.holiday[self.holiday['Currency'] == currency]
        currency_holiday_dates: np.array = currency_holiday['Date'].values

        date_range: np.array = np.empty(days, dtype = 'datetime64[ns]')
        date: np.array = start_date
        for i in range(days):
            while date in currency_holiday_dates:
                date += np.timedelta64(1, 'D')

            date_range[i] = date
            date += np.timedelta64(1, 'D')

        date_range_df: pd.DataFrame = pd.DataFrame(date_range)
        date_range_df.columns = ['liquidation_date']

        return date_range_df

    def generate_cash_flow_amount(
            self, 
            notional: float, 
            days: int
        ) -> pd.DataFrame:
        """
        Computes expected cash flow amount based on notional

        Parameters
        ----------
        notional: float
            Total notional amount
        days: int
            Number of days for liquidation

        Returns
        -------
        pd.DataFrame
            DataFrame of daily notional
        """
        notional_arr: np.array = np.array([notional / days] * days)
        notional_df: pd.DataFrame = pd.DataFrame(notional_arr)
        notional_df.columns = ['Notional']

        return notional_df

    def run(
            self, 
            asset: str
        ) -> pd.DataFrame:
        """
        Computes Liquidation Schedule

        Parameters
        ----------
        asset: str
            To retrieve relevant asset details

        Returns
        -------
        pd.DataFrame
            DataFrame containing liquidation schedule for asset
        """
        start_date, days, currency, notional = self.get_asset_details(asset)
        date_range_df = self.generate_cash_flow_dates(start_date, days, currency)
        notional = self.generate_cash_flow_amount(notional, days)

        liquidation_df: pd.DataFrame = pd.concat([notional, date_range_df], axis = 1)
        liquidation_df['Asset Name'] = asset
        liquidation_df['Currency'] = currency

        liquidation_df.rename(
            columns = {
                'Notional': 'Expected cash flow Amount',
                'liquidation_date': 'Expected Cash Flow Date'},
            inplace = True
        )
        
        return liquidation_df[['Asset Name', 'Currency', 'Expected cash flow Amount', 'Expected Cash Flow Date']]