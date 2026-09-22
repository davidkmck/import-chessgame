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
        console.warn("No chessboard detected in frame.");
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
}
