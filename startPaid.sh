#!/bin/bash

# Avvia il consiglio a pagamento (default)
export COUNCIL_TYPE=paid

echo "Avvio del backend per il consiglio a pagamento..."
uv run python -m backend.main &
BACKEND_PID=$!

echo "Avvio del frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "Consiglio a pagamento avviato! Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"
echo "Visita http://localhost:5173 nel browser."
echo "Per fermare: kill $BACKEND_PID $FRONTEND_PID"