"""
Report Generator Module

This module generates PDF reports for model backtest results.
"""

import os
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# Check if reportlab is available
reportlab_available = True
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except ImportError:
    reportlab_available = False
    print("ReportLab not available. PDF reports will not be generated.")

# Define global constants
REPORTS_DIR = os.path.join("static", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Define custom styles if reportlab is available
if reportlab_available:
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white,
        alignment=1
    )

def generate_backtest_report(ticker, backtest_data=None, metrics_data=None, include_plots=True):
    """
    Generate a PDF report of backtest results for a single model.
    
    Args:
        ticker (str): The ticker symbol
        backtest_data (dict, optional): Backtest result data from the backtest endpoint
        metrics_data (dict, optional): Additional metrics data for the model
        include_plots (bool): Whether to include performance plots
        
    Returns:
        str: Path to the generated PDF file or None if reportlab is not available
    """
    # Check if reportlab is available
    if not reportlab_available:
        print(f"Cannot generate PDF report for {ticker}: ReportLab not installed")
        return None
        
    # Try to load backtest data if not provided
    if backtest_data is None:
        backtest_dir = os.path.join("static", "backtests")
        backtest_file = os.path.join(backtest_dir, f"{ticker}_backtest.json")
        
        if os.path.exists(backtest_file):
            try:
                with open(backtest_file, "r") as f:
                    backtest_data = json.load(f)
            except Exception as e:
                print(f"Error loading backtest data: {str(e)}")
                # Create minimal backtest data
                backtest_data = {
                    "ticker": ticker,
                    "algorithm": "Unknown",
                    "metrics": {},
                    "final_portfolio": 0,
                    "initial_investment": 10000,
                    "num_trades": 0,
                    "trades": [],
                    "action_counts": {}
                }
        else:
            # Create minimal backtest data
            backtest_data = {
                "ticker": ticker,
                "algorithm": "Unknown",
                "metrics": {},
                "final_portfolio": 0,
                "initial_investment": 10000,
                "num_trades": 0,
                "trades": [],
                "action_counts": {}
            }
            
    # Create PDF filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{ticker}_report.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    elements = []
    
    # Title
    elements.append(Paragraph(f"Backtest Report: {ticker}", title_style))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Performance Summary
    elements.append(Paragraph("Performance Summary", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create summary table
    metrics = backtest_data.get("metrics", {})
    summary_data = [
        ["Metric", "Value"],
        ["Algorithm", backtest_data.get("algorithm", "Unknown")],
        ["Initial Investment", f"${backtest_data.get('initial_investment', 0):,.2f}"],
        ["Final Portfolio Value", f"${backtest_data.get('final_portfolio', 0):,.2f}"],
        ["Total Return", f"{((backtest_data.get('final_portfolio', 0) / backtest_data.get('initial_investment', 1)) - 1) * 100:.2f}%"],
        ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%"],
        ["Total Trades", f"{backtest_data.get('num_trades', 0)}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Action Distribution
    elements.append(Paragraph("Trading Action Distribution", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    action_counts = backtest_data.get("action_counts", {})
    if action_counts:
        # Create table for action counts
        action_data = [
            ["Action", "Count", "Percentage"],
            ["Hold", action_counts.get("hold", 0), f"{action_counts.get('hold', 0) / sum(action_counts.values()) * 100:.1f}%"],
            ["Buy", action_counts.get("buy", 0), f"{action_counts.get('buy', 0) / sum(action_counts.values()) * 100:.1f}%"],
            ["Sell", action_counts.get("sell", 0), f"{action_counts.get('sell', 0) / sum(action_counts.values()) * 100:.1f}%"]
        ]
        
        action_table = Table(action_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(action_table)
    else:
        elements.append(Paragraph("No action distribution data available.", normal_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Sample Trades
    elements.append(Paragraph("Sample Trades", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    trades = backtest_data.get("trades", [])
    if trades:
        # Create trades table
        trades_header = ["Date", "Action", "Price", "Shares", "Value"]
        trades_data = [trades_header]
        
        # Add up to 10 trades to the table
        for trade in trades[:10]:
            trades_data.append([
                trade.get("date", ""),
                trade.get("action", ""),
                f"${trade.get('price', 0):.2f}",
                f"{trade.get('shares', 0)}",
                f"${trade.get('value', 0):.2f}"
            ])
        
        trades_table = Table(trades_data, colWidths=[1.1*inch, 1*inch, 1*inch, 0.9*inch, 1*inch])
        trades_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (2, 1), (4, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(trades_table)
    else:
        elements.append(Paragraph("No trade data available.", normal_style))
        
    # Generate PDF
    doc.build(elements)
    return pdf_path

def generate_comparison_report(models_data=None, include_plots=True):
    """
    Generate a comparison report of multiple models.
    
    Args:
        models_data (list, optional): List of dictionaries containing model data
        include_plots (bool): Whether to include comparison plots
        
    Returns:
        str: Path to the generated PDF file or None if reportlab is not available
    """
    # Check if reportlab is available
    if not reportlab_available:
        print("Cannot generate comparison report: ReportLab not installed")
        return None
        
    # Initialize models_data if not provided
    if not models_data:
        models_data = []
        
        # Scan through backtest results directory for model data
        backtest_dir = os.path.join("static", "backtests")
        if os.path.exists(backtest_dir):
            for filename in os.listdir(backtest_dir):
                if filename.endswith("_backtest.json"):
                    try:
                        with open(os.path.join(backtest_dir, filename), 'r') as f:
                            backtest_data = json.load(f)
                            models_data.append(backtest_data)
                    except Exception as e:
                        print(f"Error loading backtest data from {filename}: {str(e)}")
        
        # Check if we found any models
        if not models_data:
            print("No model data found for comparison")
            return None

    # Create PDF filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"model_comparison_{timestamp}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    # Create PDF document (landscape for comparison tables)
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter), 
                           rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=72)
    elements = []
    
    # Title
    elements.append(Paragraph("Model Comparison Report", title_style))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Performance Comparison Table
    elements.append(Paragraph("Performance Metrics Comparison", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create comparison table headers
    comparison_data = [["Metric"]]
    model_names = []
    
    # Add model names to header row
    for model in models_data:
        ticker = model.get("ticker", "Unknown")
        algorithm = model.get("algorithm", "Unknown")
        model_names.append(f"{ticker} ({algorithm})")
        comparison_data[0].append(f"{ticker} ({algorithm})")
    
    # Add rows for each performance metric
    metrics_to_show = [
        ("Total Return", lambda m: ((m.get('final_portfolio', 0) / m.get('initial_investment', 1)) - 1) * 100, "%.2f%%"),
        ("Sharpe Ratio", lambda m: m.get('metrics', {}).get('sharpe_ratio', 0), "%.2f"),
        ("Max Drawdown", lambda m: m.get('metrics', {}).get('max_drawdown', 0) * 100, "%.2f%%"),
        ("Total Trades", lambda m: m.get('num_trades', 0), "%d")
    ]
    
    for metric_name, metric_fn, format_str in metrics_to_show:
        row = [metric_name]
        for model in models_data:
            value = metric_fn(model)
            row.append(format_str % value)
        comparison_data.append(row)
    
    # Create and style the table
    col_width = 9.5 / (len(model_names) + 1)
    comparison_table = Table(comparison_data, colWidths=[col_width*1.5*inch] + [col_width*inch] * len(model_names))
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(comparison_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Table for action distributions comparison
    elements.append(Paragraph("Action Distribution Comparison", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create action distribution table
    action_comparison = [["Model", "Holds", "Buys", "Sells", "Total Actions"]]
    
    for i, model in enumerate(models_data):
        action_counts = model.get("action_counts", {})
        total_actions = sum(action_counts.values())
        
        if total_actions > 0:
            action_comparison.append([
                model_names[i],
                f"{action_counts.get('hold', 0)} ({action_counts.get('hold', 0) / total_actions * 100:.1f}%)",
                f"{action_counts.get('buy', 0)} ({action_counts.get('buy', 0) / total_actions * 100:.1f}%)",
                f"{action_counts.get('sell', 0)} ({action_counts.get('sell', 0) / total_actions * 100:.1f}%)",
                f"{total_actions}"
            ])
    
    if len(action_comparison) > 1:
        action_table = Table(action_comparison, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(action_table)
    else:
        elements.append(Paragraph("No action distribution data available for comparison.", normal_style))
    
    # Generate PDF
    doc.build(elements)
    return pdf_path

def generate_portfolio_report(portfolio_id, backtest_data=None, include_plots=True):
    """
    Generate a PDF report of backtest results for a portfolio.
    
    Args:
        portfolio_id (str): The portfolio ID
        backtest_data (dict, optional): Backtest result data from the portfolio_backtest endpoint
        include_plots (bool): Whether to include performance plots
        
    Returns:
        str: Path to the generated PDF file or None if reportlab is not available
    """
    # Check if reportlab is available
    if not reportlab_available:
        print(f"Cannot generate portfolio report for {portfolio_id}: ReportLab not installed")
        return None
        
    # Try to load backtest data if not provided
    if not backtest_data:
        backtest_dir = os.path.join("static", "backtests", "multi")
        backtest_file = os.path.join(backtest_dir, f"{portfolio_id}_backtest.json")
        
        if os.path.exists(backtest_file):
            try:
                with open(backtest_file, "r") as f:
                    backtest_data = json.load(f)
            except Exception as e:
                print(f"Error loading portfolio backtest data: {str(e)}")
                # Create minimal backtest data
                backtest_data = {
                    "portfolio_id": portfolio_id,
                    "tickers": [],
                    "algorithm": "Unknown",
                    "metrics": {},
                    "final_portfolio": 0,
                    "initial_investment": 10000,
                    "num_trades": 0,
                    "trades": [],
                    "final_allocations": {},
                    "action_counts": {}
                }
        else:
            # Create minimal backtest data
            backtest_data = {
                "portfolio_id": portfolio_id,
                "tickers": [],
                "algorithm": "Unknown",
                "metrics": {},
                "final_portfolio": 0,
                "initial_investment": 10000,
                "num_trades": 0,
                "trades": [],
                "final_allocations": {},
                "action_counts": {}
            }
    
    # Create PDF filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{portfolio_id}_report.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    elements = []
    
    # Title
    elements.append(Paragraph(f"Portfolio Backtest Report: {portfolio_id}", title_style))
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add portfolio tickers
    tickers = backtest_data.get("tickers", [])
    elements.append(Paragraph(f"Tickers: {', '.join(tickers)}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Performance Summary
    elements.append(Paragraph("Performance Summary", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create summary table
    metrics = backtest_data.get("metrics", {})
    summary_data = [
        ["Metric", "Value"],
        ["Algorithm", backtest_data.get("algorithm", "Unknown")],
        ["Initial Investment", f"${backtest_data.get('initial_investment', 0):,.2f}"],
        ["Final Portfolio Value", f"${backtest_data.get('final_portfolio', 0):,.2f}"],
        ["Total Return", f"{((backtest_data.get('final_portfolio', 0) / backtest_data.get('initial_investment', 1)) - 1) * 100:.2f}%"],
        ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%"],
        ["Total Trades", f"{backtest_data.get('num_trades', 0)}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Final Asset Allocation
    elements.append(Paragraph("Final Asset Allocation", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    allocations = backtest_data.get("final_allocations", {})
    if allocations:
        allocations_data = [["Asset", "Allocation"]]
        for asset, allocation in allocations.items():
            allocations_data.append([asset, f"{allocation * 100:.1f}%"])
        
        allocations_table = Table(allocations_data, colWidths=[2.5*inch, 2.5*inch])
        allocations_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(allocations_table)
    else:
        elements.append(Paragraph("No allocation data available.", normal_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Sample Trades
    elements.append(Paragraph("Sample Trades", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    trades = backtest_data.get("trades", [])
    if trades:
        # Create trades table
        trades_header = ["Date", "Ticker", "Action", "Price", "Shares", "Value"]
        trades_data = [trades_header]
        
        # Add up to 10 trades to the table
        for trade in trades[:10]:
            trades_data.append([
                trade.get("date", ""),
                trade.get("ticker", ""),
                trade.get("action", ""),
                f"${trade.get('price', 0):.2f}",
                f"{trade.get('shares', 0)}",
                f"${trade.get('value', 0):.2f}"
            ])
        
        trades_table = Table(trades_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
        trades_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (3, 1), (5, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(trades_table)
    else:
        elements.append(Paragraph("No trade data available.", normal_style))
        
    # Generate PDF
    doc.build(elements)
    return pdf_path 