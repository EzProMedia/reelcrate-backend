"""Persistent email outbox and attempt limits, shared by API handlers."""
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(os.environ.get('REELCRATE_DATA', '/tmp/reelcrate'))
ROOT.mkdir(parents=True, exist_ok=True)

def connect():
    db = sqlite3.connect(ROOT / 'delivery.sqlite3', timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS mail (id TEXT PRIMARY KEY, fn TEXT, args TEXT, kwargs TEXT, attempts INTEGER DEFAULT 0, due REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS limits (key TEXT PRIMARY KEY, count INTEGER, expires REAL)')
    return db

def queue_email(fn, *args, delivery_id=None, **kwargs):
    with connect() as db:
        db.execute('INSERT OR IGNORE INTO mail(id,fn,args,kwargs,due) VALUES(?,?,?,?,?)',
                   (delivery_id or str(uuid.uuid4()), fn.__name__, json.dumps(args), json.dumps(kwargs), time.time()))

def rate_limit(key, maximum=10, seconds=900):
    from fastapi import HTTPException
    now = time.time()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DELETE FROM limits WHERE expires <= ?', (now,))
        db.execute('INSERT INTO limits VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1', (key, now + seconds))
        count, expires = db.execute('SELECT count,expires FROM limits WHERE key=?', (key,)).fetchone()
    if count > maximum:
        raise HTTPException(429, 'Too many attempts. Please try again later.', headers={'Retry-After': str(max(1, int(expires-now)))})

def deliver_once():
    import email_service
    with connect() as db:
        row = db.execute('SELECT id,fn,args,kwargs,attempts FROM mail WHERE due<=? ORDER BY due LIMIT 1', (time.time(),)).fetchone()
    if not row:
        return False
    mid, name, args, kwargs, attempts = row
    try:
        ok = getattr(email_service, name)(*json.loads(args), **json.loads(kwargs))
    except Exception:
        logging.exception('Background email delivery failed')
        ok = False
    with connect() as db:
        if ok:
            # Keep a deduplication receipt without personal data or reset tokens.
            db.execute("UPDATE mail SET args='[]', kwargs='{}', due=NULL WHERE id=?", (mid,))
        else:
            db.execute('UPDATE mail SET attempts=?, due=? WHERE id=?', (attempts+1, time.time()+min(3600, 30*2**min(attempts,7)), mid))
    return True

def start_mail_worker():
    stop = threading.Event()
    def run():
        while not stop.is_set():
            try:
                if deliver_once():
                    continue
            except Exception:
                logging.exception('Email outbox unavailable')
            stop.wait(2)
    thread = threading.Thread(target=run, name='mail-outbox', daemon=True)
    thread.start()
    return stop, thread
