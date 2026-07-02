const sections = Array.from(document.querySelectorAll("main section[id]"));
const links = Array.from(document.querySelectorAll(".site-nav nav a"));

const byId = new Map(
  links
    .map((link) => [link.getAttribute("href")?.replace("#", ""), link])
    .filter(([id]) => id)
);

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => link.classList.remove("active"));
      const active = byId.get(entry.target.id);
      if (active) active.classList.add("active");
    });
  },
  { rootMargin: "-38% 0px -52% 0px", threshold: 0.01 }
);

sections.forEach((section) => observer.observe(section));
