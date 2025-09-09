#!/bin/bash

echo "🔧 Quick Fix for DFS Optimizer Docker Issues"

# Stop and clean up
echo "🛑 Stopping containers and cleaning up..."
docker compose down
docker system prune -f

# Create directories
echo "📁 Creating directories..."
mkdir -p data/input data/output logs app

# Check if we have the right files
echo "📋 Checking files..."
if [ ! -f "app/main.py" ]; then
    echo "❌ Missing app/main.py"
    echo "Please ensure you have the updated files from the artifacts"
    exit 1
fi

if [ ! -f "app/simple_optimizer.py" ]; then
    echo "❌ Missing app/simple_optimizer.py"
    echo "Please ensure you have the updated files from the artifacts"
    exit 1
fi

# Create .env file if missing
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << 'EOF'
# OpenAI API Key (optional - for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file"
fi

# Create sample data if no CSV files exist
csv_count=$(find data/input -name "*.csv" 2>/dev/null | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "📊 Creating sample data..."
    
    # Create QB data
    cat > data/input/qb.csv << 'EOF'
PLAYER NAME,TEAM,POS,PROJ PTS,SALARY,OWN_PCT
Josh Allen,BUF,QB,22.5,8500,15.2
Patrick Mahomes,KC,QB,21.8,8300,12.8
Jalen Hurts,PHI,QB,21.2,8200,14.1
Lamar Jackson,BAL,QB,20.5,8000,11.3
Dak Prescott,DAL,QB,19.8,7700,8.7
Joe Burrow,CIN,QB,20.2,7900,9.4
Justin Herbert,LAC,QB,19.5,7600,7.2
Tua Tagovailoa,MIA,QB,18.8,7300,6.1
EOF

    # Create RB data
    cat > data/input/rb.csv << 'EOF'
PLAYER NAME,TEAM,POS,PROJ PTS,SALARY,OWN_PCT
Christian McCaffrey,SF,RB,20.1,9000,28.5
Austin Ekeler,LAC,RB,17.2,8400,22.1
Derrick Henry,TEN,RB,16.8,6800,15.7
Saquon Barkley,NYG,RB,16.2,8200,19.3
Jonathan Taylor,IND,RB,15.8,7900,16.2
Josh Jacobs,LV,RB,15.2,7500,12.8
Tony Pollard,DAL,RB,14.5,6600,11.4
Najee Harris,PIT,RB,13.8,6200,9.7
Kenneth Walker,SEA,RB,14.2,6900,13.5
Breece Hall,NYJ,RB,13.5,6700,10.8
James Cook,BUF,RB,12.8,6400,8.9
Aaron Jones,GB,RB,12.2,6100,7.6
EOF

    # Create WR data
    cat > data/input/wr.csv << 'EOF'
PLAYER NAME,TEAM,POS,PROJ PTS,SALARY,OWN_PCT
Tyreek Hill,MIA,WR,17.5,8800,24.3
Stefon Diggs,BUF,WR,16.8,8600,21.7
Justin Jefferson,MIN,WR,17.2,8900,26.1
Ja'Marr Chase,CIN,WR,16.5,8700,23.4
CeeDee Lamb,DAL,WR,15.8,8400,19.8
A.J. Brown,PHI,WR,15.2,8200,18.2
Davante Adams,LV,WR,14.8,8000,16.5
Cooper Kupp,LAR,WR,14.2,7800,15.1
Amon-Ra St. Brown,DET,WR,13.8,7600,14.3
Chris Olave,NO,WR,13.2,7200,12.7
DK Metcalf,SEA,WR,12.8,7000,11.4
Mike Evans,TB,WR,13.5,7400,13.9
DeVonta Smith,PHI,WR,12.2,6800,10.8
Jaylen Waddle,MIA,WR,11.8,6600,9.7
Calvin Ridley,JAX,WR,11.2,6400,8.5
Tee Higgins,CIN,WR,10.8,6200,7.8
Amari Cooper,CLE,WR,10.2,5800,6.9
Terry McLaurin,WAS,WR,9.8,5600,6.2
Michael Pittman,IND,WR,9.5,5400,5.8
Chris Godwin,TB,WR,10.5,6000,7.4
EOF

    # Create TE data
    cat > data/input/te.csv << 'EOF'
PLAYER NAME,TEAM,POS,PROJ PTS,SALARY,OWN_PCT
Travis Kelce,KC,TE,15.3,7000,18.7
Mark Andrews,BAL,TE,13.8,6500,15.2
T.J. Hockenson,MIN,TE,10.5,5900,11.8
George Kittle,SF,TE,10.2,6200,12.4
Dallas Goedert,PHI,TE,9.5,5600,9.3
Darren Waller,NYG,TE,8.8,5400,8.1
Kyle Pitts,ATL,TE,8.2,5200,7.2
Evan Engram,JAX,TE,7.8,5000,6.5
EOF

    # Create DST data
    cat > data/input/dst.csv << 'EOF'
PLAYER NAME,TEAM,POS,PROJ PTS,SALARY,OWN_PCT
Buffalo,BUF,DST,9.2,4800,16.5
San Francisco,SF,DST,8.8,4600,14.2
Dallas,DAL,DST,8.5,4400,12.8
New England,NE,DST,8.2,4200,11.3
Baltimore,BAL,DST,7.8,4000,9.7
Philadelphia,PHI,DST,7.5,3800,8.4
Denver,DEN,DST,7.2,3600,7.1
Cincinnati,CIN,DST,6.8,3400,5.9
EOF

    echo "✅ Sample data created"
fi

# Test Python locally first
echo "🐍 Testing Python components locally..."
python3 -c "
import sys
sys.path.append('app')
try:
    from simple_optimizer import SimpleDFSOptimizer
    from working_ai_analyzer import SimpleAIAnalyzer
    from data_ingestion import load_data_from_input_dir
    print('✅ All Python imports work')
except Exception as e:
    print(f'❌ Python import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python components have issues. Please check the file contents."
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t dfs-optimizer:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed. Check Dockerfile syntax."
    exit 1
fi

# Start services
echo "🚀 Starting Docker services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Test the API
echo "🧪 Testing API endpoints..."

# Test health endpoint
health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$health_response" = "200" ]; then
    echo "✅ Health endpoint working"
else
    echo "❌ Health endpoint failed (HTTP $health_response)"
    echo "📋 Checking container logs..."
    docker compose logs app
fi

# Test optimization endpoint
echo "🎯 Testing optimization endpoint..."
opt_response=$(curl -s -X POST http://localhost:8000/optimize \
    -H "Content-Type: application/json" \
    -d '{"salary_cap": 50000, "optimization_type": "cash"}' \
    -w "%{http_code}")

if [[ "$opt_response" == *"200"* ]] || [[ "$opt_response" == *"lineup"* ]]; then
    echo "✅ Optimization endpoint working"
else
    echo "❌ Optimization endpoint failed"
    echo "Response: $opt_response"
    echo "📋 Checking container logs..."
    docker compose logs app
fi

# Show running services
echo "📊 Running services:"
docker compose ps

# Show logs
echo "📋 Recent logs:"
docker compose logs --tail=20 app

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📚 Available endpoints:"
echo "   • Health: http://localhost:8000/health"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Optimize: POST http://localhost:8000/optimize"
echo "   • Players: GET http://localhost:8000/players"
echo ""
echo "🛠️  To monitor:"
echo "   • Logs: docker compose logs -f app"
echo "   • Stats: docker stats"
echo "   • Stop: docker compose down"
echo ""
echo "📁 File locations:"
echo "   • Input data: ./data/input/"
echo "   • Output: ./data/output/"
echo "   • Logs: ./logs/"
