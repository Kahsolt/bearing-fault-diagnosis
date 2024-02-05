@ECHO OFF

CD /D %~dp0
IF NOT EXIST CWRU MKDIR CWRU

PUSHD CWRU

REM Normal Baseline Data
wget -nc https://engineering.case.edu/sites/default/files/97.mat -O Normal_0.mat
wget -nc https://engineering.case.edu/sites/default/files/98.mat -O Normal_1.mat
wget -nc https://engineering.case.edu/sites/default/files/99.mat -O Normal_2.mat
wget -nc https://engineering.case.edu/sites/default/files/100.mat -O Normal_3.mat

REM 12k Drive End Bearing Fault Data (Centered @6:00 for Outer Race)

REM 0.007"
wget -nc https://engineering.case.edu/sites/default/files/105.mat -O IR007_0.mat
wget -nc https://engineering.case.edu/sites/default/files/106.mat -O IR007_1.mat
wget -nc https://engineering.case.edu/sites/default/files/107.mat -O IR007_2.mat
wget -nc https://engineering.case.edu/sites/default/files/108.mat -O IR007_3.mat

wget -nc https://engineering.case.edu/sites/default/files/130.mat -O OR007_0.mat
wget -nc https://engineering.case.edu/sites/default/files/131.mat -O OR007_1.mat
wget -nc https://engineering.case.edu/sites/default/files/132.mat -O OR007_2.mat
wget -nc https://engineering.case.edu/sites/default/files/133.mat -O OR007_3.mat

wget -nc https://engineering.case.edu/sites/default/files/118.mat -O B007_0.mat
wget -nc https://engineering.case.edu/sites/default/files/119.mat -O B007_1.mat
wget -nc https://engineering.case.edu/sites/default/files/120.mat -O B007_2.mat
wget -nc https://engineering.case.edu/sites/default/files/121.mat -O B007_3.mat

REM 0.014"
wget -nc https://engineering.case.edu/sites/default/files/169.mat -O IR014_0.mat
wget -nc https://engineering.case.edu/sites/default/files/170.mat -O IR014_1.mat
wget -nc https://engineering.case.edu/sites/default/files/171.mat -O IR014_2.mat
wget -nc https://engineering.case.edu/sites/default/files/172.mat -O IR014_3.mat

wget -nc https://engineering.case.edu/sites/default/files/197.mat -O OR014_0.mat
wget -nc https://engineering.case.edu/sites/default/files/198.mat -O OR014_1.mat
wget -nc https://engineering.case.edu/sites/default/files/199.mat -O OR014_2.mat
wget -nc https://engineering.case.edu/sites/default/files/200.mat -O OR014_3.mat

wget -nc https://engineering.case.edu/sites/default/files/185.mat -O B014_0.mat
wget -nc https://engineering.case.edu/sites/default/files/186.mat -O B014_1.mat
wget -nc https://engineering.case.edu/sites/default/files/187.mat -O B014_2.mat
wget -nc https://engineering.case.edu/sites/default/files/188.mat -O B014_3.mat

REM 0.021"
wget -nc https://engineering.case.edu/sites/default/files/209.mat -O IR021_0.mat
wget -nc https://engineering.case.edu/sites/default/files/210.mat -O IR021_1.mat
wget -nc https://engineering.case.edu/sites/default/files/211.mat -O IR021_2.mat
wget -nc https://engineering.case.edu/sites/default/files/212.mat -O IR021_3.mat

wget -nc https://engineering.case.edu/sites/default/files/234.mat -O OR021_0.mat
wget -nc https://engineering.case.edu/sites/default/files/235.mat -O OR021_1.mat
wget -nc https://engineering.case.edu/sites/default/files/236.mat -O OR021_2.mat
wget -nc https://engineering.case.edu/sites/default/files/237.mat -O OR021_3.mat

wget -nc https://engineering.case.edu/sites/default/files/222.mat -O B021_0.mat
wget -nc https://engineering.case.edu/sites/default/files/223.mat -O B021_1.mat
wget -nc https://engineering.case.edu/sites/default/files/224.mat -O B021_2.mat
wget -nc https://engineering.case.edu/sites/default/files/225.mat -O B021_3.mat

POPD
