#!/bin/bash

# Create sample QB data
cat << 'CSV' > data/input/qb.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Josh Allen,BUF,@MIA,22.5,8500,15-20%
Patrick Mahomes,KC,LV,21.8,8300,12-18%
Jalen Hurts,PHI,NYG,21.2,8200,10-15%
Lamar Jackson,BAL,CLE,20.5,8000,8-12%
Dak Prescott,DAL,@WAS,19.8,7700,6-10%
CSV

# Create sample RB data
cat << 'CSV' > data/input/rb.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Christian McCaffrey,SF,SEA,18.5,9000,25-30%
Austin Ekeler,LAC,@DEN,16.2,7500,18-22%
Tony Pollard,DAL,@WAS,15.8,7200,15-20%
Bijan Robinson,ATL,TB,15.5,7000,12-18%
Jonathan Taylor,IND,HOU,14.8,6800,10-15%
Najee Harris,PIT,@CIN,13.5,6500,8-12%
CSV

# Create sample WR data
cat << 'CSV' > data/input/wr.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Tyreek Hill,MIA,BUF,17.2,9200,20-25%
CeeDee Lamb,DAL,@WAS,16.8,8800,18-22%
Justin Jefferson,MIN,@GB,16.5,8700,15-20%
A.J. Brown,PHI,NYG,15.8,8400,12-18%
Stefon Diggs,BUF,@MIA,15.2,8200,10-15%
Davante Adams,LV,@KC,14.5,7900,8-12%
Chris Olave,NO,@CAR,13.8,7500,6-10%
CSV

# Create sample TE data
cat << 'CSV' > data/input/te.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Travis Kelce,KC,LV,14.5,7800,15-20%
T.J. Hockenson,MIN,@GB,11.2,6200,10-15%
Mark Andrews,BAL,CLE,10.8,6000,8-12%
Dallas Goedert,PHI,NYG,10.5,5800,6-10%
George Kittle,SF,SEA,10.2,5700,5-8%
CSV

# Create sample DST data
cat << 'CSV' > data/input/dst.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
San Francisco 49ers,SF,SEA,9.5,5000,12-18%
Buffalo Bills,BUF,@MIA,9.2,4800,10-15%
Dallas Cowboys,DAL,@WAS,8.8,4600,8-12%
Baltimore Ravens,BAL,CLE,8.5,4500,6-10%
Philadelphia Eagles,PHI,NYG,8.2,4400,5-8%
CSV

echo "Sample data created in data/input/"
