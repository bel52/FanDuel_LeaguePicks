#!/bin/bash

echo "Creating enhanced sample data for DFS optimizer..."

# Create sample QB data
cat << 'CSV' > data/input/qb.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Josh Allen,BUF,@MIA,22.5,8500,15-20%
Patrick Mahomes,KC,LV,21.8,8300,12-18%
Jalen Hurts,PHI,NYG,21.2,8200,10-15%
Lamar Jackson,BAL,CLE,20.5,8000,8-12%
Dak Prescott,DAL,@WAS,19.8,7700,6-10%
Joe Burrow,CIN,@CLE,21.2,8000,10-15%
Kyler Murray,ARI,@NO,19.3,7700,5-8%
Trevor Lawrence,JAC,CAR,18.2,7000,3-6%
Tua Tagovailoa,MIA,BUF,16.4,7300,2-5%
Jayden Daniels,WAS,NYG,22.1,8500,8-12%
CSV

# Create sample RB data
cat << 'CSV' > data/input/rb.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Christian McCaffrey,SF,SEA,18.5,9000,25-30%
Saquon Barkley,PHI,DAL,22.2,8400,20-25%
Bijan Robinson,ATL,TB,17.6,8800,15-20%
Jonathan Taylor,IND,HOU,16.1,8300,12-18%
Derrick Henry,BAL,CLE,16.2,8900,10-15%
Josh Jacobs,GB,DET,14.8,7800,8-12%
Austin Ekeler,WAS,NYG,9.5,5200,5-8%
Tony Pollard,TEN,@DEN,12.5,6000,6-10%
Chase Brown,CIN,@CLE,16.0,6900,18-22%
Ashton Jeanty,LV,@NE,15.0,6400,25-30%
Chuba Hubbard,CAR,@JAC,12.9,6500,8-12%
James Cook,BUF,@MIA,12.8,6800,6-10%
De'Von Achane,MIA,BUF,15.1,8200,10-15%
Bucky Irving,TB,@ATL,15.3,7700,12-18%
Jahmyr Gibbs,DET,@GB,16.5,8700,15-20%
CSV

# Create sample WR data
cat << 'CSV' > data/input/wr.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Ja'Marr Chase,CIN,@CLE,17.8,9200,20-25%
CeeDee Lamb,DAL,@WAS,15.0,8000,15-20%
Tyreek Hill,MIA,BUF,12.6,7600,8-12%
A.J. Brown,PHI,DAL,13.7,8200,10-15%
Justin Jefferson,MIN,@CHI,16.1,8800,18-22%
Amon-Ra St. Brown,DET,@GB,14.0,8400,12-18%
Drake London,ATL,TB,14.1,7000,10-15%
Malik Nabers,NYG,@WAS,14.8,7800,8-12%
Nico Collins,HOU,@LAR,14.9,7900,6-10%
Brian Thomas Jr.,JAC,CAR,14.5,7700,5-8%
Tee Higgins,CIN,@CLE,12.9,7100,8-12%
Terry McLaurin,WAS,NYG,12.0,7300,6-10%
Mike Evans,TB,@ATL,13.3,7500,10-15%
Ladd McConkey,LAC,KC,13.0,7600,8-12%
Puka Nacua,LAR,HOU,14.0,8100,15-20%
Stefon Diggs,NE,LV,9.9,6100,5-8%
Tetairoa McMillan,CAR,@JAC,11.8,5600,12-18%
Marvin Harrison Jr.,ARI,@NO,12.1,6400,8-12%
DeVonta Smith,PHI,DAL,11.2,6600,6-10%
Jaylen Waddle,MIA,BUF,10.9,5500,4-8%
CSV

# Create sample TE data
cat << 'CSV' > data/input/te.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Travis Kelce,KC,LV,10.3,6100,15-20%
Trey McBride,ARI,@NO,11.7,6300,12-18%
Brock Bowers,LV,@NE,12.3,7000,18-22%
George Kittle,SF,SEA,11.4,6500,10-15%
Sam LaPorta,DET,@GB,9.2,5900,8-12%
Mark Andrews,BAL,CLE,8.5,5500,6-10%
T.J. Hockenson,MIN,@CHI,8.5,5600,5-8%
Dallas Goedert,PHI,DAL,7.4,5400,4-8%
Evan Engram,JAC,CAR,7.7,5300,3-6%
Kyle Pitts,ATL,TB,6.9,5200,3-6%
David Njoku,CLE,CIN,9.9,5700,8-12%
Hunter Henry,NE,LV,7.1,5000,2-5%
CSV

# Create sample DST data
cat << 'CSV' > data/input/dst.csv
PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Denver Broncos,DEN,TEN,8.2,4800,12-18%
San Francisco 49ers,SF,SEA,7.6,4400,10-15%
Arizona Cardinals,ARI,@NO,8.1,4200,8-12%
Pittsburgh Steelers,PIT,@NYJ,8.1,4600,6-10%
Philadelphia Eagles,PHI,DAL,8.1,4700,15-20%
Washington Commanders,WAS,NYG,7.7,4700,5-8%
Minnesota Vikings,MIN,@CHI,8.0,4100,4-8%
Cincinnati Bengals,CIN,@CLE,6.9,4500,8-12%
Buffalo Bills,BUF,@MIA,5.7,3800,3-6%
Kansas City Chiefs,KC,LV,6.7,4500,5-8%
Baltimore Ravens,BAL,CLE,5.7,4000,2-5%
New England Patriots,NE,LV,6.7,3900,8-12%
CSV

echo "Sample data created successfully in data/input/"
echo "Files created: qb.csv, rb.csv, wr.csv, te.csv, dst.csv"
