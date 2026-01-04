#!/bin/bash

# Avvia il consiglio gratuito
export COUNCIL_TYPE=free

echo "Avvio del backend per il consiglio gratuito..."
uv run python -m backend.main &
BACKEND_PID=$!

echo "Avvio del frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "Consiglio gratuito avviato! Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
echo "Visita http://localhost:5173 nel browser."
echo "Per fermare: kill $BACKEND_PID $FRONTEND_PID"