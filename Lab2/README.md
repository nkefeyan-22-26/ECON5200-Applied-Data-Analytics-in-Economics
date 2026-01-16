# Lab 2: The Illusion of Growth & The Composition Effect

## Objective

Built a Python pipeline to ingest live economic data from the Federal Reserve Economic Data (FRED) API to analyze wage stagnation and correct for statistical biases. This project demonstrates how nominal wage growth can mask underlying economic realities and how composition effects can create misleading signals during periods of labor market disruption.

## Tech Stack

- **Python** - Core programming language
- **fredapi** - Federal Reserve API client for real-time economic data ingestion
- **pandas** - Data manipulation and time-series analysis
- **matplotlib** - Statistical visualization

## Methodology

### 1. Real Wage Calculation
Fetched nominal wage data (AHETPI - Average Hourly Earnings) and Consumer Price Index (CPI) from FRED API to calculate inflation-adjusted real wages over a 50+ year period. This revealed the gap between nominal growth and actual purchasing power.

### 2. Anomaly Detection
Identified a statistical anomaly in 2020 showing an apparent spike in real wages during the pandemic—a counterintuitive result given widespread economic disruption.

### 3. Composition Effect Correction
Fetched the Employment Cost Index (ECIWAG) to control for compositional changes in the workforce. The ECI uses fixed-weight methodology to track wages for the same jobs over time, eliminating bias from workforce composition changes.

### 4. Comparative Visualization
Created side-by-side comparisons of standard average wage metrics versus composition-controlled ECI data, rebased to identical starting points for direct comparison.

## Key Findings

### The Money Illusion
Visualizations demonstrate that while nominal wages have grown substantially over 50 years, real purchasing power has remained essentially flat when adjusted for inflation. This illustrates the classic economic concept of "money illusion"—workers see growing paychecks but experience stagnant living standards.

### The Pandemic Paradox
The 2020 wage data presents an apparent paradox: average wages spiked during a period of mass unemployment and economic contraction. Analysis revealed this was a **statistical artifact caused by the Composition Effect**:

- Low-wage service workers (restaurants, retail, hospitality) disproportionately lost jobs
- High-wage professionals (tech, finance, management) largely kept employment
- The remaining workforce had a higher average wage, not because anyone got raises, but because the calculation excluded millions of lower-wage workers

**Evidence**: The Employment Cost Index, which controls for workforce composition, showed **stable growth** during 2020 with no artificial spike. This proves the standard wage metric captured a change in *who was working*, not an increase in labor compensation or demand.

## Economic Implications

This analysis demonstrates the critical importance of understanding statistical methodology when interpreting economic indicators. The 2020 "wage boom" was not a sign of worker prosperity or tight labor markets—it was a mathematical consequence of selective unemployment. Policymakers and analysts relying on uncorrected wage data would misread the state of the labor market and worker welfare during this period.
