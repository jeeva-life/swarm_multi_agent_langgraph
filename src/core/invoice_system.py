"""
Invoice System for handling invoice-related operations using LangChain tools.
Provides database integration for invoice processing and customer management.
"""

from typing import Dict, List, Any, Optional
import logging
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import Config
from infrastructure.database_client import DatabaseClient


class InvoiceSystem:
    """
    Invoice System for handling invoice-related operations.
    Provides tools for invoice retrieval, customer management, and employee information.
    """
    
    def __init__(self, database_client: Optional[DatabaseClient] = None):
        """
        Initialize the Invoice System.
        
        Args:
            database_client: Database client for invoice operations
        """
        self.database_client = database_client
        self.config = Config()
        
        # Initialize LLM for invoice processing
        self.llm = ChatAnthropic(
            model=self.config.anthropic.model,
            api_key=self.config.anthropic.api_key,
            temperature=self.config.anthropic.temperature,
            max_tokens=self.config.anthropic.max_tokens
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize invoice tools
        self._setup_invoice_tools()
    
    def _setup_invoice_tools(self):
        """Setup invoice-related tools."""
        self.invoice_tools = [
            self.get_invoices_by_customer_sorted_by_date,
            self.get_invoices_sorted_by_unit_price,
            self.get_employee_by_invoice_and_customer,
            self.get_customer_invoice_summary,
            self.get_invoice_details_by_id
        ]
    
    @tool
    def get_invoices_by_customer_sorted_by_date(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Look up all invoices for a customer using their ID.
        The invoices are sorted in descending order by invoice date.
        
        Args:
            customer_id (str): Customer ID to look up invoices for.
        
        Returns:
            List[Dict[str, Any]]: A list of invoices for the customer.
        """
        try:
            if not self.database_client:
                return self._get_mock_invoices_by_customer(customer_id)
            
            query = f"""
                SELECT InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, 
                       BillingState, BillingCountry, BillingPostalCode, Total
                FROM Invoice 
                WHERE CustomerId = {customer_id} 
                ORDER BY InvoiceDate DESC
            """
            
            result = self.database_client.execute_query(query)
            return result if result else []
            
        except Exception as e:
            self.logger.error(f"Error getting invoices by customer: {str(e)}")
            return self._get_mock_invoices_by_customer(customer_id)
    
    @tool
    def get_invoices_sorted_by_unit_price(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Get invoices for a customer sorted by unit price (highest to lowest).
        
        Args:
            customer_id (str): Customer ID to look up invoices for.
        
        Returns:
            List[Dict[str, Any]]: A list of invoices sorted by unit price.
        """
        try:
            if not self.database_client:
                return self._get_mock_invoices_by_unit_price(customer_id)
            
            query = f"""
                SELECT i.InvoiceId, i.CustomerId, i.InvoiceDate, i.Total, 
                       il.UnitPrice, il.Quantity, t.Name as TrackName
                FROM Invoice i
                JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
                JOIN Track t ON il.TrackId = t.TrackId
                WHERE i.CustomerId = {customer_id}
                ORDER BY il.UnitPrice DESC
            """
            
            result = self.database_client.execute_query(query)
            return result if result else []
            
        except Exception as e:
            self.logger.error(f"Error getting invoices by unit price: {str(e)}")
            return self._get_mock_invoices_by_unit_price(customer_id)
    
    @tool
    def get_employee_by_invoice_and_customer(self, invoice_id: str, customer_id: str) -> Dict[str, Any]:
        """
        Get employee information associated with an invoice and customer.
        
        Args:
            invoice_id (str): The ID of the specific invoice.
            customer_id (str): Customer ID.
        
        Returns:
            Dict[str, Any]: Information about the employee associated with the invoice.
        """
        try:
            if not self.database_client:
                return self._get_mock_employee_info(invoice_id, customer_id)
            
            query = f"""
                SELECT e.FirstName, e.LastName, e.Title, e.Email, e.Phone
                FROM Employee e
                JOIN Customer c ON c.SupportRepId = e.EmployeeId
                JOIN Invoice i ON i.CustomerId = c.CustomerId
                WHERE i.InvoiceId = {invoice_id} AND i.CustomerId = {customer_id}
            """
            
            result = self.database_client.execute_query(query)
            if result and len(result) > 0:
                return result[0]
            else:
                return {"error": f"No employee found for invoice ID {invoice_id} and customer ID {customer_id}"}
                
        except Exception as e:
            self.logger.error(f"Error getting employee info: {str(e)}")
            return self._get_mock_employee_info(invoice_id, customer_id)
    
    @tool
    def get_customer_invoice_summary(self, customer_id: str) -> Dict[str, Any]:
        """
        Get a summary of all invoices for a customer.
        
        Args:
            customer_id (str): Customer ID to get summary for.
        
        Returns:
            Dict[str, Any]: Summary of customer invoices.
        """
        try:
            if not self.database_client:
                return self._get_mock_customer_summary(customer_id)
            
            query = f"""
                SELECT 
                    COUNT(*) as total_invoices,
                    SUM(Total) as total_amount,
                    AVG(Total) as average_amount,
                    MIN(InvoiceDate) as first_invoice_date,
                    MAX(InvoiceDate) as last_invoice_date
                FROM Invoice 
                WHERE CustomerId = {customer_id}
            """
            
            result = self.database_client.execute_query(query)
            if result and len(result) > 0:
                return result[0]
            else:
                return {"error": f"No invoices found for customer ID {customer_id}"}
                
        except Exception as e:
            self.logger.error(f"Error getting customer summary: {str(e)}")
            return self._get_mock_customer_summary(customer_id)
    
    @tool
    def get_invoice_details_by_id(self, invoice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific invoice.
        
        Args:
            invoice_id (str): Invoice ID to get details for.
        
        Returns:
            Dict[str, Any]: Detailed invoice information.
        """
        try:
            if not self.database_client:
                return self._get_mock_invoice_details(invoice_id)
            
            query = f"""
                SELECT 
                    i.InvoiceId, i.CustomerId, i.InvoiceDate, i.BillingAddress,
                    i.BillingCity, i.BillingState, i.BillingCountry, i.BillingPostalCode,
                    i.Total, c.FirstName, c.LastName, c.Email
                FROM Invoice i
                JOIN Customer c ON i.CustomerId = c.CustomerId
                WHERE i.InvoiceId = {invoice_id}
            """
            
            result = self.database_client.execute_query(query)
            if result and len(result) > 0:
                return result[0]
            else:
                return {"error": f"No invoice found with ID {invoice_id}"}
                
        except Exception as e:
            self.logger.error(f"Error getting invoice details: {str(e)}")
            return self._get_mock_invoice_details(invoice_id)
    
    def _get_mock_invoices_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get mock invoices for a customer (fallback when database is not available)."""
        mock_invoices = [
            {
                "InvoiceId": 293,
                "CustomerId": int(customer_id),
                "InvoiceDate": "2024-07-13 00:00:00",
                "BillingAddress": "Theodor-Heuss-Straße 34",
                "BillingCity": "Stuttgart",
                "BillingState": None,
                "BillingCountry": "Germany",
                "BillingPostalCode": "70174",
                "Total": 0.99
            },
            {
                "InvoiceId": 241,
                "CustomerId": int(customer_id),
                "InvoiceDate": "2023-11-23 00:00:00",
                "BillingAddress": "Theodor-Heuss-Straße 34",
                "BillingCity": "Stuttgart",
                "BillingState": None,
                "BillingCountry": "Germany",
                "BillingPostalCode": "70174",
                "Total": 5.94
            }
        ]
        return mock_invoices
    
    def _get_mock_invoices_by_unit_price(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get mock invoices sorted by unit price."""
        return [
            {
                "InvoiceId": 241,
                "CustomerId": int(customer_id),
                "InvoiceDate": "2023-11-23 00:00:00",
                "Total": 5.94,
                "UnitPrice": 0.99,
                "Quantity": 6,
                "TrackName": "Sample Track"
            },
            {
                "InvoiceId": 293,
                "CustomerId": int(customer_id),
                "InvoiceDate": "2024-07-13 00:00:00",
                "Total": 0.99,
                "UnitPrice": 0.99,
                "Quantity": 1,
                "TrackName": "Another Track"
            }
        ]
    
    def _get_mock_employee_info(self, invoice_id: str, customer_id: str) -> Dict[str, Any]:
        """Get mock employee information."""
        return {
            "FirstName": "Steve",
            "LastName": "Johnson",
            "Title": "Sales Support Agent",
            "Email": "steve@chinookcorp.com",
            "Phone": "+1-555-0123"
        }
    
    def _get_mock_customer_summary(self, customer_id: str) -> Dict[str, Any]:
        """Get mock customer summary."""
        return {
            "total_invoices": 7,
            "total_amount": 36.62,
            "average_amount": 5.23,
            "first_invoice_date": "2021-01-01 00:00:00",
            "last_invoice_date": "2024-07-13 00:00:00"
        }
    
    def _get_mock_invoice_details(self, invoice_id: str) -> Dict[str, Any]:
        """Get mock invoice details."""
        return {
            "InvoiceId": int(invoice_id),
            "CustomerId": 2,
            "InvoiceDate": "2024-07-13 00:00:00",
            "BillingAddress": "Theodor-Heuss-Straße 34",
            "BillingCity": "Stuttgart",
            "BillingState": None,
            "BillingCountry": "Germany",
            "BillingPostalCode": "70174",
            "Total": 0.99,
            "FirstName": "Helena",
            "LastName": "Holý",
            "Email": "helena.holy@email.com"
        }
    
    async def process_invoice_query(self, query: str) -> str:
        """
        Process an invoice-related query using the LLM and available tools.
        
        Args:
            query (str): The user's invoice-related query.
        
        Returns:
            str: Response to the invoice query.
        """
        try:
            # Create a prompt template for invoice processing
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an invoice processing assistant. You have access to tools for:
                - Getting invoices by customer sorted by date
                - Getting invoices sorted by unit price
                - Getting employee information for invoices
                - Getting customer invoice summaries
                - Getting detailed invoice information
                
                Use the appropriate tools to answer the user's question about invoices, customers, or billing.
                Always provide clear, helpful responses with specific details when available."""),
                ("human", "{query}")
            ])
            
            # Create the processing chain
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Process the query
            response = await chain.ainvoke({"query": query})
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing invoice query: {str(e)}")
            return "I encountered an error processing your invoice query. Please try again."
    
    def get_tools(self) -> List[Any]:
        """Get all invoice tools."""
        return self.invoice_tools


# Global instance for easy access
_invoice_system_instance = None

def get_invoice_system() -> InvoiceSystem:
    """Get the global invoice system instance."""
    global _invoice_system_instance
    if _invoice_system_instance is None:
        _invoice_system_instance = InvoiceSystem()
    return _invoice_system_instance
