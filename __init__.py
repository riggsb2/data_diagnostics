'''Author: Brian Riggs
Version: 05
Date Modified: 20210224

Evaluates the downstream database and reports/exports areas of concern

The script aims to be an automated way to find and report errors or inconsitencies
in the downstream database. The script will run a series of checks and export a
dashboard or series of excels to enable review of the points of interest.

These checks include:
* Alignment with TAN options (Company, Facility, drop downs, etc.)
* Variable conformity to restrictions such as nullable, zero-able, type
* reporting of duplicate observations
* reporting of outlier points in numeric variables from a IQ and OOM method
* reporting of variable and columns outside predefined TINC metrics
* reporting of variable and columns outside established sense check ranges


Reports will include:
* Observations for review based on:
 * TAN alignment
 * Variable Criteria conformity
 * Majoirty of data missing (null or zero)
* Duplicate variable report
* Duplicate observation report
* Error category report
* BadData Report
* ability to add "Cleared" or "OK" errors that don't need future review

'''