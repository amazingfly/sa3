let images = [];
let currentIndex = 0;
let slideshowInterval = null;

async function loadGallery() {
    const res = await fetch('/api/images');
    const data = await res.json();
    images = data.images;
    if (images.length > 0) renderImage();
}

function renderImage() {
    const container = document.getElementById('galleryContainer');
    container.innerHTML = `<img class="img-card" src="${images[currentIndex].url}">`;
    // Preload next
    const nextIdx = (currentIndex + 1) % images.length;
    new Image().src = images[nextIdx].url;
}

async function sendAction(action) {
    const filename = images[currentIndex].filename;
    await fetch('/api/images/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action, filename})
    });
    if (action === 'delete') {
        images.splice(currentIndex, 1);
        if (images.length === 0) {
            document.getElementById('galleryContainer').innerHTML = 'No images';
        } else {
            currentIndex = currentIndex % images.length;
            renderImage();
        }
    }
}

document.getElementById('likeBtn').addEventListener('click', () => sendAction('like'));
document.getElementById('trashBtn').addEventListener('click', () => sendAction('delete'));

// Swipe detection
let touchStartX = 0;
const container = document.getElementById('galleryContainer');
container.addEventListener('touchstart', e => touchStartX = e.changedTouches[0].screenX);
container.addEventListener('touchend', e => {
    const diff = e.changedTouches[0].screenX - touchStartX;
    if (diff > 50) { currentIndex = (currentIndex - 1 + images.length) % images.length; renderImage(); }
    else if (diff < -50) { currentIndex = (currentIndex + 1) % images.length; renderImage(); }
});

// Slideshow
document.getElementById('slideshowBtn').addEventListener('click', () => {
    if (slideshowInterval) { clearInterval(slideshowInterval); slideshowInterval = null; }
    else { slideshowInterval = setInterval(() => { currentIndex = (currentIndex + 1) % images.length; renderImage(); }, 1000); }
});

loadGallery();
