// Hero picture: shrinking the dataset cuts CO₂ (the picture carries its own labels).
export function HeroArt() {
  return (
    <div className="card heroart">
      <img src="/hero.jpg" alt="Infographic: a large dataset with a heavy CO₂ cloud is shrunk by DataDiet into a small dataset with a small CO₂ cloud" width="1200" height="900" style={{ display: 'block', width: '100%', height: 'auto' }} />
    </div>
  );
}
