// A map of the 13 classes your ML model will need to learn
const pieceMap = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', 'empty'];

async function classifyPieces(squares) {
    console.log("Processing 64 squares in batches of 8 (Standard API)...");
    
    let boardState = [];
    // Standard Roboflow URL structure: subdomain / workspace / project / version ? api_key
    const targetUrl = "https://detect.roboflow.com/david-mcknight/chess-com-piece-types/1?api_key=EOfoAxwLvo0TFydOmFFF";

    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    for (let i = 0; i < squares.length; i += 8) {
        const batch = squares.slice(i, i + 8);
        
        const batchPromises = batch.map(async (squareData) => {
            const canvas = document.createElement('canvas');
            canvas.width = 80;
            canvas.height = 80;
            const ctx = canvas.getContext('2d');
            ctx.putImageData(squareData, 0, 0);

            const base64Image = canvas.toDataURL("image/jpeg").split(',')[1];

            try {
                // Send base64 image as JSON body to the standard endpoint
                const response = await fetch(targetUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        "image": base64Image
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                let piece = 'empty';

                if (data.top) {
                    piece = mapPredictionToFEN(data.top);
                } else if (Array.isArray(data.predictions) && data.predictions.length > 0) {
                    piece = mapPredictionToFEN(data.predictions[0].class);
                }
                return piece;
            } catch (e) {
                console.error("API failed for a square:", e);
                return 'empty'; 
            }
        });

        const batchResults = await Promise.all(batchPromises);
        boardState.push(...batchResults);
        
        console.log(`Processed row ${Math.floor(i/8) + 1} of 8...`);
        await delay(250); 
    }

    const fenString = generateFEN(boardState);
    console.log("Derived FEN State: ", fenString);
    alert("FEN Generated:\n" + fenString);
}

// 4. Map the Model's labels to standard FEN letters
function mapPredictionToFEN(predictedClass) {
    // IMPORTANT: You will need to change these keys to match EXACTLY 
    // what the Roboflow model outputs (e.g., some say "White_Knight", some say "N")
    const map = {
        'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B', 'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K',
        'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b', 'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k',
        'empty': 'empty', 'blank': 'empty'
    };
    
    // Fallback to 'empty' if the class isn't recognized
    return map[predictedClass.toLowerCase()] || 'empty';
}

function generateFEN(boardArray) {
    let fen = '';
    for (let row = 0; row < 8; row++) {
        let emptyCount = 0;
        for (let col = 0; col < 8; col++) {
            const piece = boardArray[row * 8 + col];
            if (piece === 'empty') {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fen += emptyCount;
                    emptyCount = 0;
                }
                fen += piece;
            }
        }
        if (emptyCount > 0) fen += emptyCount;
        if (row < 7) fen += '/';
    }
    // Adds default active color, castling, en passant, and move counts
    return fen + ' w KQkq - 0 1'; 
}


function extractChessboard() {
    let src = cv.imread('sourceCanvas');
    let gray = new cv.Mat();
    let blurred = new cv.Mat();
    let edges = new cv.Mat();
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    // Reduce noise and detect edges
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.Canny(blurred, edges, 50, 150, 3, false);

    // Find all external contours
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let boardContour = null;
    
    // Locate the largest 4-point polygon (the board)
    for (let i = 0; i < contours.size(); ++i) {
        let cnt = contours.get(i);
        let area = cv.contourArea(cnt, false);
        let peri = cv.arcLength(cnt, true);
        
        let approx = new cv.Mat();
        cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

        if (approx.rows === 4 && area > maxArea) {
            maxArea = area;
            if (boardContour !== null) boardContour.delete();
            boardContour = approx.clone();
        }
        cnt.delete(); approx.delete();
    }

    if (boardContour) {
        warpBoard(src, boardContour);
        boardContour.delete();
    } else {
        console.warn("No inner chessboard detected. Assuming the entire uploaded image is the board.");
        
        // FALLBACK: The image is likely a cropped digital screenshot. 
        // Skip the complex perspective math and just stretch it to a perfect 640x640 square.
        const boardSize = 640;
        let warped = new cv.Mat();
        cv.resize(src, warped, new cv.Size(boardSize, boardSize), 0, 0, cv.INTER_LINEAR);
        cv.imshow('flatBoardCanvas', warped);
        warped.delete();
        
        // Move to the next step: slice the board into 64 pieces
        scanFlattenedBoard();
    }

    src.delete(); gray.delete(); blurred.delete(); 
    edges.delete(); contours.delete(); hierarchy.delete();
}

function warpBoard(src, boardContour) {
    let points = [];
    for (let i = 0; i < 4; i++) {
        points.push({ 
            x: boardContour.data32S[i * 2], 
            y: boardContour.data32S[i * 2 + 1] 
        });
    }
    
    // Sort vertically to separate top and bottom corners
    points.sort((a, b) => a.y - b.y);
    
    // Sort horizontally to define left and right
    let top = points.slice(0, 2).sort((a, b) => a.x - b.x);
    let bottom = points.slice(2, 4).sort((a, b) => b.x - a.x); // BR first, then BL
    
    let orderedPoints = [top[0], top[1], bottom[0], bottom[1]];

    let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
        orderedPoints[0].x, orderedPoints[0].y,
        orderedPoints[1].x, orderedPoints[1].y,
        orderedPoints[2].x, orderedPoints[2].y,
        orderedPoints[3].x, orderedPoints[3].y
    ]);
    
    // Lock the output to a perfect multiple of 8
    const boardSize = 640; 
    let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        boardSize, 0,
        boardSize, boardSize,
        0, boardSize
    ]);

    // Apply the mathematical stretch
    let M = cv.getPerspectiveTransform(srcTri, dstTri);
    let warped = new cv.Mat();
    cv.warpPerspective(src, warped, M, new cv.Size(boardSize, boardSize), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

    // Render the flattened board to the UI
    cv.imshow('flatBoardCanvas', warped);

    srcTri.delete(); dstTri.delete(); M.delete(); warped.delete();
    
    // Move to the next step: slice the board into 64 pieces
    scanFlattenedBoard();
}

function scanFlattenedBoard() {
    const flatCanvas = document.getElementById('flatBoardCanvas');
    const ctx = flatCanvas.getContext('2d');
    
    // If you used 640 for the warp, this will be 80 pixels
    const squareSize = flatCanvas.width / 8; 
    const squares = [];

    // Loop through rows (Ranks 8 to 1) and columns (Files a to h)
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            // Extract the pixel data for this specific square
            const squareData = ctx.getImageData(col * squareSize, row * squareSize, squareSize, squareSize);
            squares.push(squareData);
        }
    }

    console.log(`Successfully sliced ${squares.length} squares. Ready for ML classification.`);
    classifyPieces(squares);
}
