## Welcome to COVID-19 Insights

Some updates and projections based on current data and models about the COVID-19 pandemic.

Evolution and prediction of the COVID-19 in [Spain](https://github.com/pablocarb/covid19/blob/master/covid-19-predictions.ipynb).

<ul>
<li>{{ site.baseurl }}</li>
  {% for post in site.posts %}
  <li>{{ post.url }}</li>
    <li>
      <a href="{{ site.baseurl | append: post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
