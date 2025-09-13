# app/exporter.py
import os
import csv
import logging

def export_lineups(lineups, filename):
    """Export given lineups to CSV in FanDuel format."""
    os.makedirs('data/exports', exist_ok=True)
    file_path = os.path.join('data/exports', filename)
    headers = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
    try:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for lineup in lineups:
                # Group players by position
                qb = [p['name'] for p in lineup if p['position'] == 'QB']
                rb = [p['name'] for p in lineup if p['position'] == 'RB']
                wr = [p['name'] for p in lineup if p['position'] == 'WR']
                te = [p['name'] for p in lineup if p['position'] == 'TE']
                dst = [p['name'] for p in lineup if p['position'] == 'DST']
                if not qb or not dst:
                    continue  # ensure required positions present
                qb_name = qb[0] if qb else ""
                rb1 = rb[0] if len(rb) > 0 else ""
                rb2 = rb[1] if len(rb) > 1 else ""
                wr1 = wr[0] if len(wr) > 0 else ""
                wr2 = wr[1] if len(wr) > 1 else ""
                wr3 = wr[2] if len(wr) > 2 else ""
                te1 = te[0] if te else ""
                # Determine flex (any extra RB/WR/TE beyond base count)
                flex = ""
                if len(rb) > 2:
                    flex = rb[2]
                elif len(wr) > 3:
                    flex = wr[3]
                elif len(te) > 1:
                    flex = te[1]
                dst_name = dst[0] if dst else ""
                writer.writerow([qb_name, rb1, rb2, wr1, wr2, wr3, te1, flex, dst_name])
        logging.info(f"Exported lineup(s) to {file_path}")
    except Exception as e:
        logging.error(f"Failed to export lineups: {e}")
