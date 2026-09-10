import pytest
from scone_memory.core.errors import InvalidInput


def test_html_alt_caption_and_explicit_description_stay_with_their_image():
    import scone_memory.ingestion.image_html as module
    html = '''<figure><img src="pikachu.png" alt="Pikachu" title="Electric Pokémon" aria-describedby="bio">
      <figcaption>Pikachu uses Thunderbolt.</figcaption></figure>
      <p id="bio">A yellow Pokémon.</p><figure><img src="other.png" alt="Bulbasaur">
      <figcaption>A grass Pokémon.</figcaption></figure><script>Pikachu is malware</script>'''
    images = module.image_contexts_from_html(html, source='catalog/characters.html')
    assert len(images) == 2
    first = images[0]
    assert first.src == 'pikachu.png'
    values = [attribute.value for attribute in first.context.attributes]
    assert 'Pikachu' in values and 'Pikachu uses Thunderbolt.' in values and 'A yellow Pokémon.' in values
    assert 'A grass Pokémon.' not in values and not any('malware' in value for value in values)
    assert all(attribute.origin == 'html' for attribute in first.context.attributes)
    assert not first.context.entities
    assert first.context.source == 'catalog/characters.html'


def test_html_keeps_custom_metadata_and_handles_multiple_occurrences():
    from scone_memory.ingestion.image_html import image_contexts_from_html
    images = image_contexts_from_html('<img src="a.png" alt="One" data-credit="Artist"><img src="a.png" alt="Two">', source='catalog')
    assert images[0].context.locator != images[1].context.locator
    assert any(attribute.name == 'data-credit' and attribute.value == 'Artist' for attribute in images[0].context.attributes)


def test_script_images_and_excess_html_are_not_ingested():
    from scone_memory.ingestion.image_html import image_contexts_from_html
    assert not image_contexts_from_html('<script><img src="bad.png" alt="bad"></script>', source='catalog')
    with pytest.raises(InvalidInput):
        image_contexts_from_html('x' * 1_000_001, source='catalog')


def test_deep_html_and_hidden_images_are_bounded():
    from scone_memory.ingestion.image_html import image_contexts_from_html
    assert not image_contexts_from_html('<div hidden><img src="a.png" alt="hidden"></div>', source='catalog')
    with pytest.raises(InvalidInput):
        image_contexts_from_html('<div>' * 300 + '<img src="a.png" alt="nested">', source='catalog')
