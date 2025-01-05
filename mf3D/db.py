import sqlite3
import numpy as np

class SfMDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        # Photos table: stores basic image information
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            photo_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            fx REAL,    
            fy REAL,    
            cx REAL,    
            cy REAL    
        )''')
        
        # Features table: stores SuperPoint features for each image
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS features (
            feature_id INTEGER PRIMARY KEY,
            photo_id INTEGER,
            keypoint BLOB,
            score BLOB,
            descriptor BLOB,
            FOREIGN KEY(photo_id) REFERENCES photos(photo_id)
        )''')
        
        # Matches table: stores feature matches between image pairs
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            photo_id_1 INTEGER,
            photo_id_2 INTEGER,
            feature_id_1 INTEGER,
            feature_id_2 INTEGER,
            FOREIGN KEY(photo_id_1) REFERENCES photos(photo_id),
            FOREIGN KEY(photo_id_2) REFERENCES photos(photo_id),
            FOREIGN KEY(feature_id_1) REFERENCES features(feature_id),
            FOREIGN KEY(feature_id_2) REFERENCES features(feature_id)
        )''') #  confidence REAL,
            
        self.conn.commit()
    
    def insert_features(self, photo_id, keypoints, scores, descriptors):
        cursor = self.conn.cursor()
        for kp, score, desc in zip(keypoints, scores, descriptors.T):
            kp_blob = sqlite3.Binary(kp.tobytes())  # Serialize single keypoint
            score_blob = sqlite3.Binary(score.tobytes())  # Serialize single descriptor
            desc_blob = sqlite3.Binary(desc.tobytes())  # Serialize single descriptor
            cursor.execute(
                'INSERT INTO features (photo_id, keypoint, score, descriptor) VALUES (?, ?, ?, ?)', 
                (photo_id, kp_blob, score_blob, desc_blob)
            )
        self.conn.commit()

    def insert_matches(self, photo_id1, photo_id2, fid1, fid2):
        assert len(fid1) == len(fid2), "Shape mismatch between matches"

        cursor = self.conn.cursor()
        for f1_id,f2_id in zip(fid1, fid2):
            cursor.execute(
                '''
                INSERT INTO matches (photo_id_1, photo_id_2, feature_id_1, feature_id_2)
                VALUES (?, ?, ?, ?)
                ''',
                (photo_id1, photo_id2, int(f1_id), int(f2_id))
            )
        self.conn.commit()

    def get_features_by_photo_id(self, photo_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT keypoint, score, descriptor FROM features WHERE photo_id = ?', (photo_id,))
        result = cursor.fetchall()
        
        if result is None:
            raise ValueError(f"No features found for photo_id {photo_id}")
        
        keypoints = []
        scores = []
        descriptors = []
        for keypoints_blob, score_blob, descriptors_blob in result:
            keypoint = np.frombuffer(keypoints_blob, dtype=np.float32)
            score = np.frombuffer(score_blob, dtype=np.float32)
            descriptor = np.frombuffer(descriptors_blob, dtype=np.float32)
            keypoints.append(keypoint)
            scores.append(score)
            descriptors.append(descriptor)
            
        return {
            'keypoints': np.array(keypoints),
            'scores': np.array(scores).T,
            'descriptors': np.array(descriptors).T,
        }