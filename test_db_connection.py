import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def check_db():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        db_url = config['db_url']
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        print("Connected to database successfully.")
        session.close()
        return True
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return False

if __name__ == "__main__":
    check_db()
